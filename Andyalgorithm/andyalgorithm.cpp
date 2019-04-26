#include "andyalgorithm.h"
bool sortcmp(int a, int b){return a<b;}

//10 逻辑分类 
///
/// 【1】p是否为空
/// 【2】p.size()是否为1
/// 【3】p[1]是否为*。
///     否==> 循环isMatch(s.substr(1), p.substr(1))
///     是==> 循环
///         判断isMatch(s, p.substr(2))为真的可能性
///         s = s.sunstr(1)
///
bool isMatch(string s, string p)
{
    if(p.empty()) return s.empty();

    else if(p.size()==1)
        return s.size()==1 && (p[0]==s[0] || p[0]=='.');

    else if(p[1] !='*')
        if(s.empty())
            return false;
        else
            return (p[0]==s[0] || p[0]=='.') && isMatch(s.substr(1),p.substr(1));

    while(!s.empty() && (s[0]==p[0] || p[0]=='.'))
    {
        if (isMatch(s,p.substr(2)))
            return true;
        s=s.substr(1);
    }
    return (isMatch(s,p.substr(2)));
}
//10

//25
ListNode* reverseOneGroup(ListNode* pre, ListNode* next) {
    ListNode *last = pre->next, *cur = last->next;
    while(cur != next) {
        last->next = cur->next;
        cur->next = pre->next;
        pre->next = cur;
        cur = last->next;
    }
    return last;
}

ListNode* reverseKGroup(ListNode* head, int k)
{
    if (k==1) return head;

    ListNode* nhead=new ListNode(-1);
    ListNode *pre, *next;
    nhead->next = next = head;
    pre = nhead;
    int num = 0;
    while(next!=NULL) {num++;next=next->next;}
    next = head;
    for(int i=1;i<num+1;i++)
    {
        next = next->next;
        if(i%k == 0)
        {
            pre = reverseOneGroup(pre,next);
        }
    }
    return nhead->next;

}//25

//30
vector<int> findSubstring(string s, vector<string>& words)
{
    vector<int> res;
    if( s.empty() || words.empty() || s.size() < words.size()*words[0].size()) return res;
    unordered_map<string, int> mwords;
    int m = words.size();
    int n = words[0].size();
    for(auto &at : words) ++mwords[at];

    for(int i=0, j ;i<=s.size() - m * n ;i++)
    {
        unordered_map<string, int> mtemp;
        for(j = 0; j< m; j++)
        {
            string str = s.substr(i+ j*n, n);
            if(mwords.find(str) == mwords.end()) break;
            mtemp[str]++;
            if(mtemp[str] >  mwords[str]) break;
        }
        if(j == m) res.push_back(i);
    }
    return res;
}

//31
void nextPermutation(vector<int>& nums)
{

    int i;
    for(i= nums.size()-1; i > 0; i--)
        if(nums[i] > nums[i-1])
        {
            for(int j =nums.size()-1; j >=i  ; j--)
                if(nums[i-1] < nums[j])
                {
                    swap(nums[i-1],nums[j]);
                    reverse(nums.begin()+i,nums.end());
                    return;
                }
        }
    reverse(nums.begin()+i,nums.end());

}

//31

//33
int search(vector<int>& nums, int target) {
    if(nums.empty())
        return -1;
    int mid,left=0,right=nums.size()-1;
    while(left<=right)
    {
        mid=(left+right)>>1;
        if(nums[mid]==target)
            return mid;
        if(nums[mid]<nums[right])
        {
            if(nums[mid]<target && target<=nums[right])
                left=mid+1;
            else
                right=mid-1;
        }
        else
        {
            if(nums[left]<=target && nums[mid]>target)
                right=mid-1;
            else
                left=mid+1;
        }
    }
    return -1;
}//33

//34
vector<int> searchRange(vector<int>& nums, int target)
{
    if(nums.empty())
        return vector<int>{-1,-1};
    if(nums.front()==target && nums.back()==target)
        return vector<int>{0,nums.size()-1};
    //    vector<int> res;
    int index=search(nums,target);
    int left=index,right=index;
    while(left >0 && nums[left-1]==target){left--;}
    while(right<(nums.size()-1) && nums[right+1]==target){right++;}
    return vector<int>{left,right};
    //    return res;
}//34

void drawDiamond(int n)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<3*abs(i-n/2);j++)
        {
            cout << " ";
        }
        for(int j=0;j<2*(n/2-abs(i-n/2));j++)
            cout << "*  ";
        cout << "*"<<endl;
    }
}

//36
bool isValidSudoku(vector<vector<char>>& board) {

    //int colums=border[0].size();//值为9
    //int rows=border.size();//值为9
    set<char> sr;
    for(int i=0;i<9;i++) {
        sr.clear();
        for(int j=0;j<9;j++) {
            if(board[i][j]!='.')
                sr.insert(board[i][j]);
        }
        if(sr.size()<9) return false;
    }
    for(int j=0;j<9;j++) {
        sr.clear();
        for(int i=0;i<9;i++) {
            if(board[i][j]!='.')
                sr.insert(board[i][j]);
        }
        if(sr.size()<9) return false;
    }

    for(int i=0;i<9;i+=3)
    {
        for(int j=0;j<3;j+=3)        {
            sr.clear();
            for(int k=0;k<3;k++)            {
                for(int l=0;l<3;l++){
                    if(board[i+k][j+l]!='.')
                        sr.insert(board[i+k][j+l]);
                }
            }
            if(sr.size()<9) return false;
        }
    }
    return true;

}//36

bool isValid(vector<vector<char>>& board,int& i,int& j)
{
    set<char> s;
    int count=0;
    for(int l=0;l<9;l++) {
        if(board[i][l]!='.'){
            s.insert(board[i][l]);
            count++;
        }
    }
    if(s.size()<count) return false;

    s.clear();
    count=0;
    for(int k=0;k<9;k++) {
        if(board[k][j]!='.') {
            s.insert(board[k][j]);
            count++;
        }
    }
    if(s.size()<count) return false;

    s.clear();
    count=0;
    for(int l=0;l<3;l++)//
        for(int k=0;k<3;k++){
            if(board[i/3*3+l][j/3*3+k]!='.') {
                s.insert(board[i/3*3+l][j/3*3+k]);
                count++;
            }
        }
    if(s.size()<count) return false;
    return true;
}

bool solveSudokuDFS(vector<vector<char>>& board,int i,int j)
{
    if(i==9) return true;
    else if(j==9) return solveSudokuDFS(board,i+1,0);
    if(board[i][j]=='.')
        for(int m=0;m<9;m++)
        {
            board[i][j]=m+'1';
            if(isValid(board,i,j))
                if(solveSudokuDFS(board,i,j+1))
                    return true;
            board[i][j]='.';
        }
    else
        return solveSudokuDFS(board,i,j+1);
    return false;
}



//37
void solveSudoku(vector<vector<char>>& board) {
    //利用逐行递归求解
    if(board.size()==0 || board[0].size()==0)
        return;
    solveSudokuDFS(board,0,0);
}//37

//38
string countAndSay(int n) {
    if(n<1) return "";
    string res="1",temp="1";
    while(--n)
    {
        res.clear();
        int count=1;
        for(unsigned int i=0,index=i+1;index<=temp.length();index++)
        {
            if(temp[i]==temp[index]) count++;
            else
            {
                res+=(count+'0');
                res+=temp[i];
                i=index;
                count=1;
            }
        }
        temp=res;
    }
    return res;
}//38

void combinationSumDFS(vector<int>& candidate,int target,int start,vector<int>& out,vector<vector<int>>& res)
{
    if(target<0) return;
    else if(target==0) {res.push_back(out); return;}
    for(unsigned int i=start;i<candidate.size();i++)
    {
        if(i>start && candidate[i]==candidate[i-1]) continue;
        out.push_back(candidate[i]);
        combinationSumDFS(candidate,target-candidate[i],i+1,out,res);
        out.pop_back();
    }
}
//40
vector<vector<int>> combinationSum(vector<int>& candidates, int target)
{
    if(candidates.empty()) return vector<vector<int>>{};
    //首先进行排序
    std::sort(candidates.begin(),candidates.end(),sortcmp);

    vector<vector<int>> res;
    vector<int> out;
    combinationSumDFS(candidates,target,0,out,res);
    return res;
}//40

//44
//基本版，数据量大无法计算结果
bool isMatch2(string s, string p)
{
    if(p.empty()) return s.empty();
    if(s.empty())
    {
        if(p[0] == '*') return isMatch2(s,p.substr(1));
        else return false;
    }
    while(p[0]!='*')
    {
        if(s[0]!=p[0] && p[0]!='?') return false;
        if(s.empty() || p.empty()) break;
        s=s.substr(1);
        p=p.substr(1);
    }
    while(!p.empty() && !s.empty())
    {
        if(p.size()>1 && p[0] =='*' && p[1] =='*')
        {
            p = p.substr(1);
            continue;
        }
        if(isMatch2(s,p.substr(1))) return true;
        s = s.substr(1);
    }
    return isMatch2(s,p);
}//44

//44
////升级版，利用窗口扫描法
bool isMatch2pro(string s, string p)
{
    int i=0,j=0;
    int istart=-1,jstart=-1;
    while(j<s.size())
    {
        if(p.empty()) return false;
        else if(p[i] == s[j] || p[i] == '?')
        {
            i++;
            j++;
        }
        else if(p[i] == '*')
        {
            istart = i;
            jstart = j;
            i++;
        }
        else if(istart>-1)
        {
            i= istart +1;//神来之笔
            j = ++jstart ;
        }
        else
            return false;
    }
    while(i<p.size() && p[i]=='*') i++;
    return i==p.size();
}//44

//45
void jumpDFS(vector<int>& nums,int n,int count,int res)
{
    if(n+1>=nums.size() && count<res){res=count; return;}
    for(int i=n;i<nums.size();i++)
    {
        for(int j=0;j<nums[i];j++)
        {
            count++;
            if(count>=res) break;
            else jumpDFS(nums,n+j,count,res);
            count--;
        }
    }
}
int jump(vector<int>& nums)
{
    if(nums.empty()) return 0;
    int res;
    jumpDFS(nums,0,nums.size(),res);
    return res;
}//45

void permuteDFS(vector<int>& nums,int index,vector<int>& visited,vector<int>& out,vector<vector<int>>& res)
{
    if(index==nums.size()) {res.push_back(out);return;}
    for(int i=0;i<nums.size();i++)
    {
        if(visited[i]==0)
        {
            visited[i]=1;//nums中第i个数据被注册掉
            out.push_back(nums[i]);
            permuteDFS(nums,index+1,visited,out,res);
            out.pop_back();
            visited[i]=0;
        }
    }
}
//46
vector<vector<int>> permute(vector<int>& nums) {
    if(nums.empty()) return vector<vector<int>>{};
    vector<vector<int>> res;
    vector<int> visited(nums.size(),0);
    vector<int> out;
    permuteDFS(nums,0,visited,out,res);
    return res;
}//46

//49
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    map<string,vector<string>> m;
    vector<vector<string>> res;
    for(auto at:strs)
    {
        vector<int> vec(26,0);
        string temp="";
        for(auto ch:at)
            vec[ch-'a']++;
        for(auto i:vec) temp+=to_string(i);
        m[temp].push_back(at);
    }

    for(auto at:m)
        res.push_back(at.second);
    return res;
}//49


//51 递归算法
bool isValid(vector<string>& res,int &rows, int& cols)
{
    int size = res.size();
    //判断列
    for(int i=0;i<rows;i++)
        if(res[i][cols] == 'Q') return false;
    //判断左上斜线
    for(int i=rows-1,j=cols-1;i>=0 && j>=0;i--,j-- )
        if(res[i][j] == 'Q') return false;
    //判断右上斜线
    for(int i=rows-1,j=cols+1;i>=0 && j<size;i--,j++ )
        if(res[i][j] == 'Q') return false;
    return true;
}

void solveNQueensHelper(vector<vector<string>>& res,vector<string>& queens, int curr)
{
    int n = queens.size();
    if(n==curr)
    {
        res.push_back(queens);
        return;
    }
    for(int i=0;i<n;i++)
    {
        if(isValid(queens, curr,i))
        {
            queens[curr][i] = 'Q';
            solveNQueensHelper(res,queens,curr+1);
            queens[curr][i] ='.';
        }
    }
}

vector<vector<string>> solveNQueens(int n)
{
    vector<vector<string>> res{};
    //表示生成n行string，每个string由n个.组成
    vector<string> queens(n,string(n,'.'));
    solveNQueensHelper(res,queens,0);
    return res;

}//51

//52
int totalNQueens(int n)
{
    vector<vector<string>> res;
    vector<string> queens(n, string(n,'.'));
    solveNQueensHelper(res,queens, 0);
    return res.size();
}//52

//53
//普通算法
int maxSubArray(vector<int>& nums)
{
    int res=INT_MIN;//设定为最小值
    int sum=0;
    for(auto& at: nums)  // 如果使用foreach(auto at, nums)，需要添加头文件“std::algorithm.h”
    {
        sum +=at;
        if(sum < at) sum=at;
        if(sum > res) res = sum;
    }
    return res;
}

//分治算法 （--还不大懂--）
int maxSubArrayHelper(vector<int>& nums, int left,int right)
{
    if (left >= right) return nums[left];
    int mid = left + (right - left) / 2;
    int lmax = maxSubArrayHelper(nums, left, mid - 1);
    int rmax = maxSubArrayHelper(nums, mid + 1, right);
    int mmax = nums[mid], t = mmax;
    for (int i = mid - 1; i >= left; --i) {
        t += nums[i];
        mmax = max(mmax, t);
    }
    t = mmax;
    for (int i = mid + 1; i <= right; ++i) {
        t += nums[i];
        mmax = max(mmax, t);
    }
    return max(mmax, max(lmax, rmax));

}
int maxSubArray2(vector<int>& nums)
{
    if(nums.empty()) return 0;
    maxSubArrayHelper(nums,0,nums.size()-1);
}

//53

//54
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    vector<int> res;
    if (matrix.empty() || matrix[0].empty()) return res;
    int m = matrix.size(), n = matrix[0].size();
    int c = m > n ? (n + 1) / 2 : (m + 1) / 2;
    int p = m, q = n;
    for (int i = 0; i < c; ++i, p -= 2, q -= 2) {
        for (int col = i; col < i + q; ++col)
            res.push_back(matrix[i][col]);
        for (int row = i + 1; row < i + p; ++row)
            res.push_back(matrix[row][i + q - 1]);
        if (p == 1 || q == 1) break;
        for (int col = i + q - 2; col >= i; --col)
            res.push_back(matrix[i + p - 1][col]);
        for (int row = i + p - 2; row > i; --row)
            res.push_back(matrix[row][i]);
    }
    return res;

}//54

//55
bool canJump1(vector<int>& nums) {
    vector<int> dp(nums.size(),0);
    for(int i=1;i<nums.size();i++)
        dp[i]=max(dp[i-1],nums[i-1])-1;
    for(auto at:dp)
        if(at<0)
            return false;
    return true;
} //动态规划法
bool canJump(vector<int>& nums)
{
    int reach=0;
    for(int i=0;i<nums.size();i++)
    {
        if(i>reach) return false;
        else if(reach+1>=nums.size()) return true;
        reach=max(nums[i]+i,reach);
    }
    return true;

}//贪婪算法
//55

//57
vector<Interval> insert(vector<Interval>& intervals, Interval newInterval) {
    vector<Interval> res;
    int cur=0;
    for(int i=0;i<intervals.size();i++)
    {
        if(intervals[i].start>newInterval.end)
        {
            res.push_back(intervals[i]);
        }
        else if(intervals[i].end<newInterval.start)
        {
            res.push_back(intervals[i]);
            cur++;
        }
        else
        {
            newInterval.start=min(newInterval.start,intervals[i].start);
            newInterval.end=max(newInterval.end,intervals[i].end);
        }
    }
    res.insert(res.begin()+cur,newInterval);
    return res;
}//57

//59
//需要判断n为奇偶的情况
vector<vector<int>> generateMatrix(int n)
{
    vector<vector<int>> res(n,vector<int>(n,0));
    int left=0,right=n-1;
    int up=0,down=n-1;
    int val = 1;
    for(int i=0;i<n/2;i++)
    {
        for(int j=left;j<=right;j++)
            res[i][j] = val++;
        up++;
        for(int j=up; j<=down;j++)
            res[j][right] = val++;
        right--;
        for(int j=right;j>=left;j--)
            res[down][j] = val++;
        down--;
        for(int j=down;j>=up;j--)
            res[j][left] = val++;
        left++;
    }
    if(n%2) res[n/2][n/2] = n*n;
    return res;
}//59

//==60
string getPermutation(int n, int k) {
    string num="123456789";
    vector<int> direc(n,1);
    string res;
    for(int i=1;i<n;i++)
    {
        direc[i]=direc[i-1]*i;
    }
    k--;
    int j;
    for(int i=n;i>0;i--)
    {
        j=k/direc[i-1];
        res.push_back(num[j]);
        k=k%direc[i-1];
        num.erase(j,1);
    }

    return res;
}//60

//61
ListNode* rotateRight(ListNode* head, int k) {
    if(head==NULL) return head;
    ListNode* temp=head;
    ListNode* start;
    int n=1;
    while(temp->next!=NULL)
    {
        n++;
        temp=temp->next;
    }

    temp->next=head;
    for(int i=0;i<n-k%n;i++)
        temp=temp->next;
    start=temp->next;
    temp->next=NULL;
    return start;
}//61

//62
int uniquePaths(int m, int n){
    int small=std::min(m-1,n-1);
    int big=std::max(m-1,n-1);
    int total=m+n-2;
    long long ded=1;
    long long div=1;
    for(int i=1;i<=small;i++)
    {
        ded*=total;
        div*=i;
        total--;
    }
    return ded/div;

}//62

//63
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
{
    if(obstacleGrid.empty() || obstacleGrid[0].empty() || obstacleGrid[0][0]==1)
        return 0;
    int rows = obstacleGrid.size();
    int cols = obstacleGrid[0].size();
    vector<vector<long>> dp(rows+1,vector<long>(cols+1,0));//有时候值太大，int无法存储
    dp[0][1]=1;

    for(int i=1;i<=rows;i++)
        for(int j=1;j<=cols;j++)
        {
            if(obstacleGrid[i-1][j-1]==1) continue;
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    return dp[rows][cols];
}//63

//64
int minPathSum(vector<vector<int>>& grid)
{
    if(grid.empty() || grid[0].empty()) return 0;
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<long>> dp(m,vector<long>(n,0));
    dp[0][0]=grid[0][0];
    for(int i=1;i<m;i++) dp[i][0]=dp[i-1][0]+grid[i][0];
    for(int i=1;i<n;i++) dp[0][i]=dp[0][i-1]+grid[0][i];
    for(int i=1;i<m;i++)
        for(int j=1;j<n;j++)
            dp[i][j] = grid[i][j] + min(dp[i][j-1],dp[i-1][j]);
    return dp[m-1][n-1];
}//64

//65
bool isNumber(string s)
{
    bool num=false, dot=false,exp=false,sign=false,numAfterE=true;
    for(int i=0;i<s.size();i++)
    {
        if(s[i]==' ')
        {
            if(i+1<s.size() && s[i+1]!=' ' && (num || dot || exp || sign)) return false;
            continue;
        }
        if(s[i]>='0' && s[i]<='9')
        {
            num = true;
            numAfterE=true;
            continue;
        }
        if(s[i]=='-' || s[i]=='+')
        {
            if( i>0 && s[i-1]!=' ' && !(s[i-1]=='e' || s[i-1]=='E')) return false;
            sign=true;
            continue;
        }
        if(s[i]=='.')
        {
            if(dot || exp) return false;
            dot = true;
            continue;
        }
        if(s[i]=='e' || s[i]=='E')
        {
            if(!num || exp) return false;
            exp=true;
            numAfterE=false;
            continue;
        }
        else return false;
    }
    return num&&numAfterE;
}//65;

//68
vector<string> fullJustify(vector<string>& words, int maxWidth)
{
    vector<string> res;
    vector<int> len;
    int num=-1,sum=-1;
    int i, curr;
    for(string str: words)
    {
        if(str.size() > maxWidth) return res;
        len.push_back(str.size());//记录各字符串长度
    }

    for(i=0,curr=i;i<words.size();i++)
    {

        int d,p;
        if(sum+len[i]+1 <= maxWidth)
        {
            sum= sum+len[i]+1;
            num++;// n+1个有效字符
            continue;
        }
        string line{};
        line.insert(line.begin(),words[curr].begin(),words[curr].end());

        if(num==0)
            line.insert(line.end(),maxWidth-len[curr],' ');

        else
        {
            d = (maxWidth-sum)/num;//字符间可以增加空格数
            p = (maxWidth-sum)%num;//前p个字符可以增加一个空格
            for(int j=0;j<num;j++)
            {
                curr++;
                if(j<p) line.insert(line.end(),d+2,' ');
                else line.insert(line.end(),d+1,' ');
                line.insert(line.end(),words[curr].begin(),words[curr].end());
            }
        }
        res.push_back(line);
        num = -1;
        sum = -1;
        swap(i,curr);
    }
    sum = -1;
    if(curr<words.size())
    {
        string line{};
        for(;curr<words.size();curr++)
        {
            sum = sum + len[curr]+1;
            line = line + " " + words[curr];

        }
        line.insert(line.end(),maxWidth-sum,' ');
        res.push_back(line.substr(1));
    }
    return res;
}//68


//69
int mySqrt(int x) {
    long res=x;
    while(res*res>x)
    {
        res=(res+x/res)/2;//至于为何，朕也不知道。。。
    }
    return res;
}//69

//70
int climbStairs(int n) {
    if(n<=1) return 1;
    vector<int> dp{1,2};
    if(n==2) return 2;
    for(int i=2;i<n;i++)
        dp.push_back(dp[i-2]+dp[i-1]);
    return dp.back();
}//70

//71
string simplifyPath(string path)
{
    vector<string> p;
    string res;
    int i=0,start;

    while(i<path.size())
    {
        while(i<path.size() && path[i]=='/') i++;//斜杠全部可以取消，不保留
        start = i;
        while(i<path.size() && path[i]!='/') i++;
        string s = path.substr(start,i-start);
        start = i;
        if(s == ".." )
        {
            if(!p.empty())  p.pop_back();
            continue;
        }

        else if(s!="." && s!="") p.push_back(s);
    }
    if(p.empty()) return "/";
    for(string s: p)
        res = res+ "/" +s;
    return res;
}//71

//72

int minDistanceHelper(string& word1, int i, string& word2, int j,vector<vector<int>>& memo)
{
    if(i == word1.size()) return word2.size()-j;
    else if(j == word2.size()) return word1.size()-i;
    else if(memo[i][j] > 0) return memo[i][j];

    if(word1[i] == word2[j]) return minDistanceHelper(word1, i+1, word2, j+1, memo);
    else
    {
        int insertCnt = minDistanceHelper(word1, i, word2, j+1, memo);
        int deleteCnt= minDistanceHelper(word1,i+1, word2, j,memo);
        int replaceCnt = minDistanceHelper(word1, i+1, word2, j+1, memo);
        memo[i][j] = min(insertCnt,min(deleteCnt,replaceCnt))+1;
    }
    return memo[i][j];
}

int minDistance(string word1, string word2)
{
    int m = word1.size();
    int n = word2.size();
    vector<vector<int>> memo(m,vector<int>(n,0));
    return minDistanceHelper(word1,0,word2,0,memo);
}//72

//74
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if(matrix.empty() || matrix[0].empty()) return false;
    //一维i ==> 二维[i/n][i%n]
    int m=matrix.size();
    int n=matrix[0].size();
    int mid;
    int left=0,right=m*n-1;
    if(matrix[0][0]>target || matrix[m-1][n-1]<target) return false;
    while(left<=right)
    {
        mid=(left+right)>>1;
        if(matrix[mid/n][mid%n]>target)
            right=mid-1;
        else if(matrix[mid/n][mid%n]<target)
            left=mid+1;
        else
            return true;
    }
    return false;
}//74

//76
string minWindow(string s, string t)
{
    string res{""};
    unordered_map<char,int> letterCnt;
    int left=0, cnt =0, len=INT_MAX;

    for(char c: t) letterCnt[c]++;
    for(int i=0;i<s.size();i++)
    {
        if(--letterCnt[s[i]]>=0) cnt++;
        while(cnt==t.size())
        {
            if(len > i-left+1)
            {
                len = i - left +1;
                res = s.substr(left, len);
            }
            if(++letterCnt[s[left]] > 0) cnt--;
            left++;
        }
    }
    return res;
}//76


//77
void combineDFS(int n,int k,int level,vector<int>& out,vector<vector<int>>& res)
{
    if(out.size()==k) {res.push_back(out);return;}

    for(int i=level;i<=n;i++)
    {
        out.push_back(i);
        combineDFS(n,k,i+1,out,res);
        out.pop_back();
    }
}

vector<vector<int>> combine(int n, int k) {
    vector<int> out;
    vector<vector<int>> res;
    combineDFS(n,k,1,out,res);
    return res;

}//77

//78
void subsetsDFS(vector<int>& nums,int n,int level,vector<int>& out,vector<vector<int>>& res)
{
    if(out.size()==n){res.push_back(out);return;}
    for(int i=level;i<nums.size();i++)
    {
        out.push_back(nums[i]);
        subsetsDFS(nums,n,i+1,out,res);
        out.pop_back();
    }
}
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> out;
    for(int i=0;i<=nums.size();i++)
        subsetsDFS(nums,i,0,out,res);
    return res;

} //78

//79
bool existHelper(vector<vector<char>>& board, string word, int n)
{
    if(n == word.size()) return true;
    for(int i=n;i<word.size();i++)
    {
        if(word)
    }
}
bool exist(vector<vector<char>>& board, string word)
{

}//79

//80
int removeDuplicates(vector<int>& nums) {
    if(nums.size()<2) return nums.size();
    for(int i=2;i<nums.size();)
    {
        if(nums[i]==nums[i-1] && nums[i]==nums[i-2])
            nums.erase(nums.begin()+i);
        else
            i++;
    }
    return nums.size();
}//80

//81
bool search(vector<int>& nums, int target)
{
    if(nums.empty()) return false;
    int left=0,right=nums.size()-1;

    while(left<right)
    {
        int mid = (left+right)/2;
        if(nums[mid]==target) return true;
        else if(nums[mid]< nums[right])
        {
            if(nums[mid]< target && nums[right] >= target) left=mid+1;
            else right=mid-1;
        }
        else if(nums[mid]> nums[right])
        {
            if(nums[mid]> target && nums[left] <= target) right=mid-1;
            else left=mid+1;
        }
        else right--;//在常数情况下如2，2，2，2，2，3, 1，1，1    target=3
    }
    return nums[left]==target;
}//81

//82
ListNode* deleteDuplicates(ListNode* head) {
    if(head==NULL) return head;
    ListNode* start=new ListNode(0);
    start->next=head;
    ListNode* temp=start;
    while(temp->next!=NULL)
    {
        ListNode* cur=temp->next;
        while(cur->next!=NULL && cur->val==cur->next->val) cur=cur->next;
        if(cur!=temp->next) temp->next=cur->next;
        else temp=temp->next;
    }
    return start->next;
}//82

//84
int largestRectangleArea(vector<int> &height)
{
    int res=0;
    stack<int> s;
    height.push_back(0);//in order to accumulate the last number: height[n]
    for(int i=0;i<(signed)height.size();i++)
    {
        if(s.empty() || height[i]>height[s.top()]) s.push(i);
        else
        {
            int curr = s.top(); s.pop();
            res = max(res,height[curr]* (s.empty()? i:i-s.top()-1));
            i--;
        }
    }
    return res;
}//84

int maximalRectangleHelper(vector<int> height)
{
    int res = 0;
    height.push_back(0);
    stack<int> s;
    for(int i=0;i<(signed)height.size();i++)
    {
        if(s.empty() || height[s.top()] < height[i] ) s.push(i);
        else
        {
            int curr = s.top();
            s.pop();
            res = max(res, height[curr] * (s.empty()?i: i-s.top()-1));
            i--;
        }
    }
    return res;
}
//85
int maximalRectangle(vector<vector<char>>& matrix)
{
    if(matrix.empty() || matrix[0].empty()) return 0;
    int m = matrix.size(), n = matrix[0].size();
    vector<int> height(n,0);
    int res=0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            height[j] = matrix[i][j]=='0' ? 0:height[j]+1;
        res = max(res,maximalRectangleHelper(height));
    }
    return res;
}//85

//87
bool isScramble(string s1, string s2)
{
    if(s1 == s2) return true;
    if(s1.size()!=s2.size()) return false;
    string str1 = s1, str2 = s2;
    sort(str1.begin(),str1.end());
    sort(str2.begin(),str2.end());
    if(str1 != str2) return false;
    for(int i=1;i<(signed)s1.size();i++)
    {
        string s11 = s1.substr(0,i);
        string s12 = s1.substr(i);
        string s21 = s2.substr(0,i);
        string s22 = s2.substr(i);
        if(isScramble(s11,s21) && isScramble(s12,s22)) return true;
        s21 = s2.substr(s2.size()-i,i);
        s22 = s2.substr(0,s2.size()-i);
        if(isScramble(s11,s21) && isScramble(s12,s22)) return true;
    }
    return false;
}

/*
The purpose of this function is to convert an unsigned
binary number to reflected binary Gray code.

The operator >> is shift right. The operator ^ is exclusive or.  ----wikipedia
*/
unsigned int binaryToGray(unsigned int num)
{
    return (num >> 1) ^ num;
}

/*
The purpose of this function is to convert a reflected binary
Gray code number to a binary number. ----wikipedia
*/
unsigned int grayToBinary(unsigned int num)
{
    unsigned int mask;
    for (mask = num >> 1; mask != 0; mask = mask >> 1)
    {
        num = num ^ mask;//yi或
    }
    return num;
}
//89
vector<int> grayCode(int n)
{
    if(n == 0) return vector<int>{0};
    vector<int> res;
    int num=1;
    for(int i=0;i<n;i++) num*=2;
    for(int i=0;i<num;i++)
        res.push_back(binaryToGray(i));
    return res;
}//89

void subsetsWithDupHelper(vector<int>& nums, vector<int>& out,int pos, vector<vector<int>>& res)
{
    res.push_back(out);
    for(unsigned i=pos;i<nums.size();i++)
    {
        out.push_back(nums[i]);
        subsetsWithDupHelper(nums,out,i+1,res);
        out.pop_back();
        while(i+1<nums.size() && nums[i]==nums[i+1]) i++;
    }
}
//90
vector<vector<int>> subsetsWithDup(vector<int>& nums)
{
    vector<vector<int>> res;
    vector<int> out;
    if(nums.empty()) return res;
    sort(nums.begin(), nums.end());
    subsetsWithDupHelper(nums,out,0,res);


    return res;
}//90

void numDecodingsHelper(string s, char& out, int pos, int res)
{
    res++;
    for
}

//91
int numDecodings(string s)
{
    if(s.empty()) return 0;
    int res=0;
    char out;
    for(size_t i=0;i<s.size();i++)
        if(s[i]>='0' && s[i]<='9') return 0;//check for excluding '0'~'9' character
    res = numDecodingsHelper(s,out,0,res);
    return res;
}


