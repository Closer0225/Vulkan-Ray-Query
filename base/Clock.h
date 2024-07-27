#ifndef CLOCK_H
#define CLOCK_H

#ifdef _WIN32
#include <Windows.h>
#else
#include <chrono>
#endif

#ifdef _MSC_VER
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include<string>
class DLLEXPORT Clock
{
public:
	Clock();
	~Clock();
	void begin();
	void end();
	float getInterval() const;//Interval 间隔,应该指时间间隔
	void displayInterval(const std::string& name) const;

private:
#ifdef _WIN32
	LARGE_INTEGER frequency;
	LARGE_INTEGER counterBegin;
	LARGE_INTEGER counterEnd;
	double interval;
#else
	std::chrono::system_clock::time_point timeBegin;
	std::chrono::system_clock::time_point timeEnd;
	std::chrono::duration<float, std::milli> timeInterval;
#endif
};

#endif 
