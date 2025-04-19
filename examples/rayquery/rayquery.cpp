#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"
#include "VulkanRaytracingSample.h"
#include "Clock.h"
#define ENABLE_VALIDATION false

constexpr int numTimestamps = 20;

class VulkanExample : public VulkanRaytracingSample
{
public:
	glm::vec3 lightPos = glm::vec3();
	float timeStampPeriod = 1e-6f;

	struct UniformData {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::vec3 lightPos;
	} uniformData;
	vks::Buffer ubo;

	vkglTF::PointModel scene;

	VkPhysicalDeviceScalarBlockLayoutFeatures enabledScalarBlockLayoutFeatures{};

	//graphics
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;
	// compute Resources for the compute part of the example
	struct {
		vks::Buffer points;
		vks::Buffer queries;
		vks::Buffer sortqueries;
		vks::Buffer normals;
		vks::Buffer morton_in;
		vks::Buffer morton_out;
		vks::Buffer histogram;
		vks::Buffer maxpoint;
		vks::Buffer minpoint;
		VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		VkFence fence;								// Synchronization fence to avoid rewriting compute CB if still in use
		struct {
			VkDescriptorSet raytracing;
			VkDescriptorSet multiRadixSortHistograms;
			VkDescriptorSet multiRadixSortHistograms2;
			VkDescriptorSet multiRadixSort;
			VkDescriptorSet multiRadixSort2;
		} descriptorSets;
		struct {
			VkDescriptorSetLayout raytracing;
			VkDescriptorSetLayout multiRadixSortHistograms;
			VkDescriptorSetLayout multiRadixSortHistograms2;
			VkDescriptorSetLayout multiRadixSort;
			VkDescriptorSetLayout multiRadixSort2;
		} descriptorSetLayouts;
		struct {
			VkPipelineLayout raytracing;
			VkPipelineLayout multiRadixSortHistograms;
			VkPipelineLayout multiRadixSortHistograms2;
			VkPipelineLayout multiRadixSort;
			VkPipelineLayout multiRadixSort2;
		} pipelineLayouts;
		std::array<VkPipeline, 5> pipelines;
		struct {
			uint32_t g_num_elements;
			uint32_t shift; 
			uint32_t num_workgroups;
			uint32_t num_blocks_per_workgroup = 32;
		} pushConstants;
	} compute;
	float radius = 5.0;
	VulkanRaytracingSample::AccelerationStructure bottomLevelAS{};
	VulkanRaytracingSample::AccelerationStructure topLevelAS{};

	VkPhysicalDeviceRayQueryFeaturesKHR enabledRayQueryFeatures{};

	VkQueryPool queryPool = VK_NULL_HANDLE;
	uint64_t timestamps[numTimestamps * 3] = { 0 };

	VulkanExample() : VulkanRaytracingSample()
	{
		title = "Ray queries for ray traced";
		camera.type = Camera::CameraType::lookat;
		timerSpeed *= 0.25f;
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
		camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 3.0f, -10.0f));
		rayQueryOnly = true;
		enableExtensions();
		enabledDeviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	}

	~VulkanExample()
	{
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		ubo.destroy();
		compute.morton_in.destroy();
		compute.morton_out.destroy();
		compute.histogram.destroy();
		compute.maxpoint.destroy();
		compute.minpoint.destroy();
		compute.points.destroy();
		compute.normals.destroy();
		compute.queries.destroy();
		compute.sortqueries.destroy();	
		vkDestroyPipelineLayout(device, compute.pipelineLayouts.multiRadixSortHistograms, nullptr);
		vkDestroyPipelineLayout(device, compute.pipelineLayouts.multiRadixSort, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayouts.multiRadixSortHistograms, nullptr);
		vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayouts.multiRadixSort, nullptr);
		for (auto i = 0; i < 3; ++i)
			vkDestroyPipeline(device, compute.pipelines[i], nullptr);
		deleteAccelerationStructure(bottomLevelAS);
		deleteAccelerationStructure(topLevelAS);
		vkDestroyFence(device, compute.fence, nullptr);
		vkDestroyCommandPool(device, compute.commandPool, nullptr);
	}

	// Setup and fill the compute shader storage buffers containing primitives for the raytraced scene
	void prepareStorageBuffers()
	{
		vkglTF::memoryPropertyFlags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		const std::string& fileName = "../assets/lucy.asc";
		scene.numpoint = 0;
		std::vector<glm::vec3>points;
		std::vector<glm::vec3>normals;
		std::vector<int>initindex;
		std::vector<vkglTF::PointModel::Vertert>verterts;
		std::vector<vkglTF::PointModel::Dimensions>AABBs;
		FILE* fp = fopen(fileName.c_str(), "r");
		glm::vec3 point;
		glm::vec3 normal;
		glm::vec3 localminpoint= glm::vec3(FLT_MAX);
		glm::vec3 localmaxpoint= glm::vec3(-FLT_MAX);
		char line[1024];
		while (fgets(line, 1023, fp))
		{
			sscanf(line, "%f%f%f", &point.x, &point.y, &point.z);
			if (point.x > localmaxpoint.x)
				localmaxpoint.x = point.x;
			if (point.y > localmaxpoint.y)
				localmaxpoint.y = point.y;
			if (point.z > localmaxpoint.z)
				localmaxpoint.z = point.z;
			if (point.x < localminpoint.x)
				localminpoint.x = point.x;
			if (point.y < localminpoint.y)
				localminpoint.y = point.y;
			if (point.z < localminpoint.z)
				localminpoint.z = point.z;
			vkglTF::PointModel::Vertert vertert;
			vertert.pos = glm::vec3(point.x, point.y, point.z);
			points.push_back(glm::vec3(point.x, point.y, point.z));
			normals.push_back(glm::vec3(0, 0, 0));
			verterts.push_back(vertert);
			vkglTF::PointModel::Dimensions dimensions;
			dimensions.max = glm::vec3(point.x + radius, point.y + radius, point.z + radius);
			dimensions.min = glm::vec3(point.x - radius, point.y - radius, point.z - radius);
			AABBs.push_back(dimensions);
			initindex.push_back(scene.numpoint);
			scene.numpoint++;
		}
		fclose(fp);
		scene.loadPointCloud(verterts, AABBs, vulkanDevice, queue);
		VkDeviceSize storageBufferSize = scene.numpoint * sizeof(glm::vec3);
		vks::Buffer stagingBuffer;
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			storageBufferSize,
			points.data());
		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.points,
			storageBufferSize);
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.points.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			storageBufferSize,
			points.data());
		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.queries,
			storageBufferSize);
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.queries.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			storageBufferSize,
			normals.data());
		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.normals,
			storageBufferSize);
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.normals.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.sortqueries,
			storageBufferSize);

		std::vector<glm::vec3>minpoints;
		std::vector<glm::vec3>maxpoints;
		minpoints.push_back(localminpoint);
		maxpoints.push_back(localmaxpoint);
		storageBufferSize = 1 * sizeof(glm::vec3);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			storageBufferSize,
			minpoints.data());
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.minpoint,
			1 * sizeof(glm::vec3));
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.minpoint.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			storageBufferSize,
			maxpoints.data());
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.maxpoint,
			1 * sizeof(glm::vec3));
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion.size = storageBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, compute.maxpoint.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		storageBufferSize = scene.numpoint * sizeof(uint32_t);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
			&compute.morton_in,
			storageBufferSize);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.morton_out,
			storageBufferSize);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&compute.histogram,
			storageBufferSize);
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Scene vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&ubo,
			sizeof(UniformData)));

		// Map persistent
		VK_CHECK_RESULT(ubo.map());

		updateLight();
		updateUniformBuffers();
	}

	void updateLight()
	{
		// Animate the light source
		lightPos.x = cos(glm::radians(timer * 360.0f)) * 40.0f;
		lightPos.y = -50.0f + sin(glm::radians(timer * 360.0f)) * 20.0f;
		lightPos.z = 25.0f + sin(glm::radians(timer * 360.0f)) * 5.0f;
	}

	void updateUniformBuffers()
	{
		uniformData.projection = camera.matrices.perspective;
		uniformData.view = camera.matrices.view;
		uniformData.model = glm::mat4(1.0f);
		uniformData.lightPos = lightPos;
		memcpy(ubo.mapped, &uniformData, sizeof(UniformData));
	}

	void setupDescriptorSetLayout()
	{
		// 定义描述符集合布局descriptorSetLayout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
			// Binding 1: Acceleration structure
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));
		// 创建管线布局pipelineLayout
		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));
	}

	void preparePipelines()
	{
		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
		pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
		pipelineCI.pRasterizationState = &rasterizationStateCI;
		pipelineCI.pColorBlendState = &colorBlendStateCI;
		pipelineCI.pMultisampleState = &multisampleStateCI;
		pipelineCI.pViewportState = &viewportStateCI;
		pipelineCI.pDepthStencilState = &depthStencilStateCI;
		pipelineCI.pDynamicState = &dynamicStateCI;
		pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCI.pStages = shaderStages.data();

		// Scene rendering with ray traced shadows applied
		pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ vkglTF::VertexComponent::Position });
		rasterizationStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
		shaderStages[0] = loadShader(getShadersPath() + "rayquery/scene.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "rayquery/scene.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipeline));
	}

	/*
		Create the bottom level acceleration structure contains the scene's actual geometry (vertices, triangles)
	*/
	void createBottomLevelAccelerationStructure()
	{

		VkDeviceOrHostAddressConstKHR vertexsBufferDeviceAddress{};
		VkDeviceOrHostAddressConstKHR aabbsBufferDeviceAddress{};

		vertexsBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.vertices.vertert.buffer);
		aabbsBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.aabbs.aabb.buffer);

		uint32_t numpoint = static_cast<uint32_t>(scene.numpoint);
		//Build
		VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
		accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
		accelerationStructureGeometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
		accelerationStructureGeometry.geometry.aabbs.data = aabbsBufferDeviceAddress;
		accelerationStructureGeometry.geometry.aabbs.stride = sizeof(vkglTF::PointModel::Dimensions);

		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationStructureBuildGeometryInfo.geometryCount = 1;
		accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&numpoint,
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructure(bottomLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, accelerationStructureBuildSizesInfo);

		// Create a small scratch buffer used during build of the bottom level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.dstAccelerationStructure = bottomLevelAS.handle;
		accelerationBuildGeometryInfo.geometryCount = 1;
		accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
		accelerationStructureBuildRangeInfo.primitiveCount = numpoint;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		deleteScratchBuffer(scratchBuffer);
	}

	/*
		The top level acceleration structure contains the scene's object instances
	*/
	void createTopLevelAccelerationStructure()
	{
		VkTransformMatrixKHR transformMatrix = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f };

		VkAccelerationStructureInstanceKHR instance{};
		instance.transform = transformMatrix;
		instance.instanceCustomIndex = 0;
		instance.mask = 0xFF;
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.accelerationStructureReference = bottomLevelAS.deviceAddress;

		// Buffer for instance data
		vks::Buffer instancesBuffer;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&instancesBuffer,
			sizeof(VkAccelerationStructureInstanceKHR),
			&instance));

		VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
		instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instancesBuffer.buffer);

		VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
		accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationStructureBuildGeometryInfo.geometryCount = 1;
		accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

		uint32_t primitive_count = 1;

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&primitive_count,
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructure(topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, accelerationStructureBuildSizesInfo);

		// Create a small scratch buffer used during build of the top level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.dstAccelerationStructure = topLevelAS.handle;
		accelerationBuildGeometryInfo.geometryCount = 1;
		accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
		accelerationStructureBuildRangeInfo.primitiveCount = 1;
		accelerationStructureBuildRangeInfo.primitiveOffset = 0;
		accelerationStructureBuildRangeInfo.firstVertex = 0;
		accelerationStructureBuildRangeInfo.transformOffset = 0;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		deleteScratchBuffer(scratchBuffer);
		instancesBuffer.destroy();
	}

	//创建描述池descriptorPool
	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}
	//分配和更新描述池
	void setupDescriptorSets()
	{
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;

		// Debug display
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

		// Scene rendering with shadow map applied
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		writeDescriptorSets = {
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &ubo.descriptor)
		};

		VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = vks::initializers::writeDescriptorSetAccelerationStructureKHR();
		descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
		descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;

		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
		accelerationStructureWrite.dstSet = descriptorSet;
		accelerationStructureWrite.dstBinding = 1;
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

		writeDescriptorSets.push_back(accelerationStructureWrite);
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}

	
	void buildComputeCommandBuffer()
	{
		auto computeCommandBuffer = compute.commandBuffer;
		auto addMemoryBarrier = [&computeCommandBuffer](VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask)
			{
				VkMemoryBarrier memoryBarrier = vks::initializers::memoryBarrier();
				memoryBarrier.srcAccessMask = srcAccessMask;
				memoryBarrier.dstAccessMask = dstAccessMask;
				vkCmdPipelineBarrier(computeCommandBuffer, srcStageMask, dstStageMask, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
			};

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));
		vkCmdResetQueryPool(compute.commandBuffer, queryPool, 0, 2);
		
		compute.pushConstants.shift = 0; compute.pushConstants.g_num_elements = scene.numpoint; compute.pushConstants.num_workgroups = (scene.numpoint + 256 - 1) / 256;
		vkCmdPushConstants(compute.commandBuffer, compute.pipelineLayouts.multiRadixSortHistograms, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[1]);
		vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSortHistograms, 0, 1, &compute.descriptorSets.multiRadixSortHistograms, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSort, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[2]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSort, 0, 1, &compute.descriptorSets.multiRadixSort, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 2);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 3);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		compute.pushConstants.shift = 8;
		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSortHistograms2, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[3]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSortHistograms2, 0, 1, &compute.descriptorSets.multiRadixSortHistograms2, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 4);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 5);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSort2, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[4]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSort2, 0, 1, &compute.descriptorSets.multiRadixSort2, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 6);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 7);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		compute.pushConstants.shift = 16;
		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSortHistograms, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[1]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSortHistograms, 0, 1, &compute.descriptorSets.multiRadixSortHistograms, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 8);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 9);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSort, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[2]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSort, 0, 1, &compute.descriptorSets.multiRadixSort, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 10);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 11);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		compute.pushConstants.shift = 24;
		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSortHistograms2, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[3]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSortHistograms2, 0, 1, &compute.descriptorSets.multiRadixSortHistograms2, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 12);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 13);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdPushConstants(computeCommandBuffer, compute.pipelineLayouts.multiRadixSort2, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants), &compute.pushConstants);
		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[4]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.multiRadixSort2, 0, 1, &compute.descriptorSets.multiRadixSort2, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 14);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 256 - 1) / 256, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 15);
		addMemoryBarrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelines[0]);
		vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayouts.raytracing, 0, 1, &compute.descriptorSets.raytracing, 0, 0);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 16);
		vkCmdDispatch(compute.commandBuffer, (scene.numpoint + 1024 - 1) / 1024, 1, 1);
		vkCmdWriteTimestamp(compute.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 17);
		vkEndCommandBuffer(compute.commandBuffer);
	}

	// Prepare the compute pipeline that generates the ray traced image
	void prepareCompute()
	{
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.pNext = NULL;
		queueCreateInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		queueCreateInfo.queueCount = 1;
		vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

		//创建描述池descriptorPool
		std::vector<VkDescriptorPoolSize> poolSizes = {
				vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24),
				vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 5);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		// 定义raytracing描述符集合布局compute.descriptorSetLayout
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_SHADER_STAGE_COMPUTE_BIT,0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,VK_SHADER_STAGE_COMPUTE_BIT,1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_SHADER_STAGE_COMPUTE_BIT,2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_SHADER_STAGE_COMPUTE_BIT,3),
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout =vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayouts.raytracing));
		//定义管线布局compute.pipelineLayout
		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo =vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayouts.raytracing,1);
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		pPipelineLayoutCreateInfo.pSetLayouts = &compute.descriptorSetLayouts.raytracing;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayouts.raytracing));
		//分配描述子集
		VkDescriptorSetAllocateInfo allocInfo =vks::initializers::descriptorSetAllocateInfo(descriptorPool,&compute.descriptorSetLayouts.raytracing,1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets.raytracing));
		VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = vks::initializers::writeDescriptorSetAccelerationStructureKHR();
		descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
		descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;
		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
		accelerationStructureWrite.dstSet = compute.descriptorSets.raytracing;
		accelerationStructureWrite.dstBinding = 1,
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets =
		{
			// Binding 0: points storage buffer 
			vks::initializers::writeDescriptorSet(compute.descriptorSets.raytracing,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&compute.queries.descriptor),
			accelerationStructureWrite,
			// Binding 2: normals storage buffer 
			vks::initializers::writeDescriptorSet(compute.descriptorSets.raytracing,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2,&compute.normals.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.raytracing,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3,&compute.points.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);
		
		//pre multiRadixSortHistograms
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
		};
		descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayouts.multiRadixSortHistograms));
		VkPushConstantRange pushConstantRanges = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants) };
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayouts.multiRadixSortHistograms, 1);
		pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayouts.multiRadixSortHistograms));
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayouts.multiRadixSortHistograms, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets.multiRadixSortHistograms));
		computeWriteDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &compute.morton_in.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &compute.histogram.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &compute.queries.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &compute.maxpoint.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &compute.minpoint.descriptor),
		};
		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
		
		//pre multiRadixSortHistograms2
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
		};
		descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayouts.multiRadixSortHistograms2));
		pushConstantRanges = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(compute.pushConstants) };
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayouts.multiRadixSortHistograms2, 1);
		pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayouts.multiRadixSortHistograms2));
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayouts.multiRadixSortHistograms2, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets.multiRadixSortHistograms2));
		computeWriteDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &compute.morton_out.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &compute.histogram.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &compute.queries.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &compute.maxpoint.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSortHistograms2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &compute.minpoint.descriptor),
		};
		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);

		//pre multiRadixSort
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
		};
		descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayouts.multiRadixSort));
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayouts.multiRadixSort, 1);
		pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayouts.multiRadixSort));
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayouts.multiRadixSort, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets.multiRadixSort));
		computeWriteDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &compute.morton_in.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &compute.morton_out.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &compute.histogram.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &compute.queries.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &compute.sortqueries.descriptor),
		};
		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
		
		//pre multiRadixSort2
		setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
		};
		descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayouts.multiRadixSort2));
		pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayouts.multiRadixSort2, 1);
		pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRanges;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &compute.pipelineLayouts.multiRadixSort2));
		allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayouts.multiRadixSort2, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets.multiRadixSort2));
		computeWriteDescriptorSets =
		{
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &compute.morton_out.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &compute.morton_in.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &compute.histogram.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &compute.sortqueries.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSets.multiRadixSort2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &compute.queries.descriptor),
		};
		vkUpdateDescriptorSets(device, computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, NULL);
		// Create compute shader pipelines raytracing
		VkComputePipelineCreateInfo computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayouts.raytracing,
				0);
		auto createComputePipeline = [&](int index, const std::string& name, int subgroupSize = 0)
			{
				computePipelineCreateInfo.stage = loadShader("./../shaders/glsl/rayquery/" + name + ".comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
				if (subgroupSize > 0)
				{
					VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeCreateInfo = {};
					subgroupSizeCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
					subgroupSizeCreateInfo.requiredSubgroupSize = subgroupSize;
					computePipelineCreateInfo.stage.pNext = &subgroupSizeCreateInfo;
					computePipelineCreateInfo.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
				}
				VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipelines[index]));
			};
		
		createComputePipeline(0, "raytracing");

		computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayouts.multiRadixSortHistograms,
				0);
		createComputePipeline(1, "multi_radixsort_histograms");



		computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayouts.multiRadixSort,
				0);

		createComputePipeline(2, "multi_radixsort", 32);

		computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayouts.multiRadixSortHistograms2,
				0);
		createComputePipeline(3, "multi_radixsort_histograms");

		computePipelineCreateInfo =
			vks::initializers::computePipelineCreateInfo(
				compute.pipelineLayouts.multiRadixSort2,
				0);

		createComputePipeline(4, "multi_radixsort", 32);
		// 创建命令池->在命令池中分配命令缓冲区->为命令缓冲区赋值->创建栅栏->提交命令缓冲区队列开始计算任务->使用栅栏等待计算结束
		// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations
		VkCommandBufferAllocateInfo cmdBufAllocateInfo =
			vks::initializers::commandBufferAllocateInfo(
				compute.commandPool,
				VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				1);

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &compute.commandBuffer));

		// Fence for compute CB sync
		VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &compute.fence));

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		VkViewport viewport;
		VkRect2D scissor;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };;
			clearValues[1].depthStencil = { 1.0f, 0 };

			VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
			renderPassBeginInfo.renderPass = renderPass;
			renderPassBeginInfo.framebuffer = frameBuffers[i];
			renderPassBeginInfo.renderArea.extent.width = width;
			renderPassBeginInfo.renderArea.extent.height = height;
			renderPassBeginInfo.clearValueCount = 2;
			renderPassBeginInfo.pClearValues = clearValues;

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			// 3D scene
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
			scene.draw(drawCmdBuffers[i], 0, pipeline);

			VulkanExampleBase::drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void getEnabledFeatures()
	{
		// Enable features required for ray tracing using feature chaining via pNext		
		enabledBufferDeviceAddresFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		enabledBufferDeviceAddresFeatures.bufferDeviceAddress = VK_TRUE;
		enabledBufferDeviceAddresFeatures.pNext = nullptr;

		//enabledRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
		//enabledRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
		//enabledRayTracingPipelineFeatures.pNext = &enabledBufferDeviceAddresFeatures;

		enabledScalarBlockLayoutFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES;
		enabledScalarBlockLayoutFeatures.scalarBlockLayout = VK_TRUE;
		enabledScalarBlockLayoutFeatures.pNext = &enabledBufferDeviceAddresFeatures;

		enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
		enabledAccelerationStructureFeatures.pNext = &enabledScalarBlockLayoutFeatures;

		enabledRayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
		enabledRayQueryFeatures.rayQuery = VK_TRUE;
		enabledRayQueryFeatures.pNext = &enabledAccelerationStructureFeatures;


		deviceCreatepNextChain = &enabledRayQueryFeatures;

		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		timeStampPeriod = properties.limits.timestampPeriod / 1e6f;

		VkPhysicalDeviceSubgroupProperties subgroupProperties{};
		subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

		VkPhysicalDeviceProperties2 physicalDeviceProperties{};
		physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		physicalDeviceProperties.pNext = &subgroupProperties;

		vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);
	}

	void draw()
	{

		vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &compute.fence);

		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;

		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, compute.fence));

		VulkanExampleBase::prepareFrame();

		// Command buffer to be submitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	// Setup a query pool for storing pipeline statistics
	void setupQueryPool()
	{
		VkQueryPoolCreateInfo queryPoolInfo = {};
		queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryPoolInfo.queryCount = numTimestamps;
		VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolInfo, NULL, &queryPool));

		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdResetQueryPool(commandBuffer, queryPool, 0, queryPoolInfo.queryCount);
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);
	}

	// Retrieves the results of the pipeline statistics query submitted to the command buffer
	void getQueryResults()
	{
		// We use vkGetQueryResults to copy the results into a host visible buffer
		vkGetQueryPoolResults(
			device,
			queryPool,
			0,
			numTimestamps,
			sizeof(timestamps),
			timestamps,
			sizeof(uint64_t),
			VK_QUERY_RESULT_64_BIT);
	}

	void prepare()
	{	// 定义描述集合布局->创建管线布局->创建描述池->分配和更新描述符集
		VulkanRaytracingSample::prepare();
		setupQueryPool();
		prepareUniformBuffers();
		prepareStorageBuffers();
		setupDescriptorSetLayout();
		createBottomLevelAccelerationStructure();
		createTopLevelAccelerationStructure();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSets();
		prepareCompute();
		buildCommandBuffers();
		prepared = true;
	}

	int frequency = 0;
	virtual void render()
	{
		if (!prepared)
			return;

		draw();
		vkWaitForFences(device, 1, &compute.fence, VK_TRUE, UINT64_MAX);
		getQueryResults();
		for (int i = 1,shift=0; i < 16; i +=2,shift+=8) {
			printf("%d:  %f\n", shift,(timestamps[i] - timestamps[i-1]) * timeStampPeriod);
		}
		printf("GPU:  %f\n",(timestamps[17] - timestamps[16]) * timeStampPeriod);
		if (!paused || camera.updated)
		{
			updateLight();
			updateUniformBuffers();
		}
		frequency++;
		if (frequency == 10)
			getresult();
	}

	void getresult() {
		vkResetFences(device, 1, &compute.fence);
		const std::string& fileName = "../assets/normal.asc";

		vks::Buffer readBuffer;
		VkDeviceSize bufferSize = scene.numpoint * sizeof(glm::vec3);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&readBuffer,
			bufferSize);

		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(copyCmd, compute.queries.buffer, readBuffer.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		glm::vec3* cpuArray = new glm::vec3[scene.numpoint];
		void* mapped;
		VK_CHECK_RESULT(vkMapMemory(device, readBuffer.memory, 0, bufferSize, 0, &mapped));
		memcpy(cpuArray, mapped, bufferSize);
		vkUnmapMemory(device, readBuffer.memory);
		readBuffer.destroy();

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&readBuffer,
			bufferSize);
		
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion = {};
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(copyCmd, compute.normals.buffer, readBuffer.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		glm::vec3* cpuindex = new glm::vec3[scene.numpoint];
		VK_CHECK_RESULT(vkMapMemory(device, readBuffer.memory, 0, bufferSize, 0, &mapped));
		memcpy(cpuindex, mapped, bufferSize);
		vkUnmapMemory(device, readBuffer.memory);
		readBuffer.destroy();

		bufferSize = scene.numpoint * sizeof(glm::uint);
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&readBuffer,
			bufferSize);
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion = {};
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(copyCmd, compute.points.buffer, readBuffer.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		glm::uint* morton = new glm::uint[scene.numpoint];
		VK_CHECK_RESULT(vkMapMemory(device, readBuffer.memory, 0, bufferSize, 0, &mapped));
		memcpy(morton, mapped, bufferSize);
		vkUnmapMemory(device, readBuffer.memory);
		readBuffer.destroy();

		FILE* fp = fopen(fileName.c_str(), "w");
		if (fp)
		{

			for (auto i = 0; i < scene.numpoint; ++i)
			{
				fprintf(fp, "%f %f %f %f %f %f %d\n", cpuArray[i].x, cpuArray[i].y, cpuArray[i].z, cpuindex[i].x, cpuindex[i].y, cpuindex[i].z, morton[i]);
			}
			fclose(fp);
		}
	}
};

VULKAN_EXAMPLE_MAIN()
//C:\VulkanSDK\1.3.261.0\Bin\glslangValidator.exe -V raytracing.comp -o raytracing.comp.spv --target-env vulkan1.3
//C:\VulkanSDK\1.3.261.0\Bin\glslangValidator.exe -V scene.vert -o scene.vert.spv --target-env vulkan1.3
//compute(matrix,tmp, vec_tmp, evecs, evals)

