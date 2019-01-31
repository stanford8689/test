#include<stdlib.h>
#include<string>
#include<string.h>
#include<stdio.h>
#include "../include/function.h"
void test(){

	const char* c;
	string s="1234";
	c = s.c_str();
	cout<<c<<endl; //输出：1234
	s="abcd";
	cout<<c<<endl; //输出：abcd

}

