#include "clock.h"

Clock::Clock()
{

}

Clock::~Clock()
{

}

void Clock::begin()
{
	//std::chrono精度不够，只能到1ms
#ifdef _WIN32
	//函数返回高精确度性能计数器的值
	QueryPerformanceCounter(&counterBegin);
	//返回每秒嘀哒声的个数
	QueryPerformanceFrequency(&frequency);
#else
	timeBegin = std::chrono::system_clock::now();
#endif
}

void Clock::end()
{
#ifdef _WIN32
	QueryPerformanceCounter(&counterEnd);
	//相当于总滴答数除以每秒滴答多少次，秒转为毫秒
	interval = 1000 * ((double)counterEnd.QuadPart - (double)counterBegin.QuadPart) / frequency.QuadPart;
#else
	timeEnd = std::chrono::system_clock::now();
	//持续的时间 -  duration
	timeInterval = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeBegin);//微秒
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
	//count()表示这段时间的长度
	printf("%s: %fms\n", name.c_str(), timeInterval.count());
#endif
}