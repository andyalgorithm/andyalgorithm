#ifndef ANDYALGORITHM_H
#define ANDYALGORITHM_H

#include "andyglobal.h"

#include <cstdlib>
#include <cmath>
#include <stack>
#include <iostream>
#include <vector>
#include <set>
#include <unordered_map>
//#include <algorithm>

#include <QDebug>
#include <QPoint>
#include <QPointF>

using namespace std;

//== 10 ==//
bool isMatch(string s, string p);

//== 25 ==//
ListNode* reverseOneGroup(ListNode* pre, ListNode* next);
ListNode* reverseKGroup(ListNode* head, int k);

//== 30 ==//
 vector<int> findSubstring(string s, vector<string>& words);

 //== 31 ==//
 void nextPermutation(vector<int>& nums);

//==  33 ==//
int search(vector<int>& nums, int target);
//== 34 ==//
vector<int> searchRange(vector<int>& nums, int target);

//== 36 ==//
bool isValidSudoku(vector<vector<char>>& board);

void drawDiamond(int n);

//== 37 ==//
void solveSudoku(vector<vector<char>>& board);

//== 38 ==//
string countAndSay(int n);

//== 40 ==// //不重复, Holy shit !
vector<vector<int>> combinationSum(vector<int>& candidates, int target);

//== 44 ==//
//缺点：执行速度慢，效率低
bool isMatch2(string s, string p);

//利用窗口扫描的方法进行确定
bool isMatch2pro(string s, string p);

//== 45 ==//
int jump(vector<int>& nums);

//== 46 ==//动态规划（回调函数）
vector<vector<int>> permute(vector<int>& nums);

//== 49 ==//
vector<vector<string>> groupAnagrams(vector<string>& strs);

//== 51 ==//
vector<vector<string>> solveNQueens(int n);

//== 52 ==//
int totalNQueens(int n);

//== 53 ==//
//普通算法
int maxSubArray(vector<int>& nums);

//分治算法
int maxSubArray2(vector<int>& nums);

//== 54 ==// 二维螺旋 格雷编码
vector<int> spiralOrder(vector<vector<int>>& matrix);

//== 55 ==//
bool canJump1(vector<int>& nums);
bool canJump2(vector<int>& nums);

//== 57 ==//
vector<Interval> insert(vector<Interval>& intervals, Interval newInterval);

//== 59 ==//
vector<vector<int>> generateMatrix(int n);


//== 60 ==//
string getPermutation(int n, int k);//利用阶乘的规律来查找

//== 61 ==//
ListNode* rotateRight(ListNode* head, int k);

//== 62 ==// 绝妙
int uniquePaths(int m, int n);

//== 63 ==//
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);

//== 64 ==//
int minPathSum(vector<vector<int>>& grid);

//== 65 ==//
bool isNumber(string s);

//== 68 ==//
vector<string> fullJustify(vector<string>& words, int maxWidth);

//== 69 ==//
int mySqrt(int x);

//== 70 ==// 斐波那契数列
int climbStairs(int n);

//== 71 ==//
string simplifyPath(string path);

//== 72 ==//
int minDistance(string word1, string word2);

//== 74 ==//
bool searchMatrix(vector<vector<int>>& matrix, int target);

//== 76 ==//
string minWindow(string s, string t);

//== 77 ==// DFS动态查询
vector<vector<int>> combine(int n, int k);

//== 78 ==//
vector<vector<int>> subsets(vector<int>& nums);//求所有子集

//== 79 ==//
bool exist(vector<vector<char>>& board, string word);

//== 80 ==//鬼斧神工之笔。本大爷手笔
int removeDuplicates(vector<int>& nums);

//== 81 ==//
bool search(vector<int>& nums, int target);

//== 82 ==//
ListNode* deleteDuplicates(ListNode* head);

//== 84 ==//
int largestRectangleArea(vector<int> &height);

//== 85 ==//
int maximalRectangle(vector<vector<char>>& matrix);

//== 87 ==//
bool isScramble(string s1, string s2);

//== 89 ==//
vector<int> grayCode(int n);

//== 90 ==//
vector<vector<int>> subsetsWithDup(vector<int>& nums);

//== 91 ==//
int numDecodings(string s);





#endif // ANDYALGORITHM_H
