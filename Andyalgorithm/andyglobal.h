#ifndef ANDYGLOBAL_H
#define ANDYGLOBAL_H
#include <cstdlib>

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct Interval {
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(int s, int e) : start(s), end(e) {}
};

struct point{
    int x;
    int y;
    point():x(0),y(0){}
    point(int m, int n): x(m),y(n){}
};



#endif // ANDYGLOBAL_H
