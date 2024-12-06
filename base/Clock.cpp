#include "clock.h"

Clock::Clock()
{

}

Clock::~Clock()
{

}

void Clock::begin()
{
	//std::chrono���Ȳ�����ֻ�ܵ�1ms
#ifdef _WIN32
	//�������ظ߾�ȷ�����ܼ�������ֵ
	QueryPerformanceCounter(&counterBegin);
	//����ÿ���������ĸ���
	QueryPerformanceFrequency(&frequency);
#else
	timeBegin = std::chrono::system_clock::now();
#endif
}

void Clock::end()
{
#ifdef _WIN32
	QueryPerformanceCounter(&counterEnd);
	//�൱���ܵδ�������ÿ��δ���ٴΣ���תΪ����
	interval = 1000 * ((double)counterEnd.QuadPart - (double)counterBegin.QuadPart) / frequency.QuadPart;
#else
	timeEnd = std::chrono::system_clock::now();
	//������ʱ�� -  duration
	timeInterval = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin);//΢��
#endif
}

float Clock::getInterval() const
{
#ifdef _WIN32
	return interval;
#else
	return timeInterval.count();
#endif
}

void Clock::displayInterval(const std::string& name) const
{
#ifdef _WIN32
	printf("%s: %lfms\n", name.c_str(), interval);
#else
	//count()��ʾ���ʱ��ĳ���
	printf("%s: %fms\n", name.c_str(), timeInterval.count());
#endif
}