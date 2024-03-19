#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <cassert>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vl vector<ll>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int n, hCnt, sCnt, visited[10005], st[10005];
vi a[105], ans;
map<string, int> Hash;

void Hashing(string s, vi& arr) {
    istringstream s2(s);
    string s3;
    
    while (getline(s2, s3, ' ')) {
        if (Hash.find(s3) == Hash.end())
            Hash.insert({ s3, hCnt++ });
        
        arr.push_back(Hash.find(s3)->second);
    }
    
    st[arr[0]] = 1;
}

void input() {
    cin >> n;
    
    cin.ignore();
    for (int i = 0; i < n; i++) {
        string s;
        getline(cin, s);
        
        Hashing(s, a[i]);
    }
    
    sCnt = hCnt;
    
    string s;
    getline(cin, s);
    
    Hashing(s, ans);
}

int f() {
    for (auto i : ans) {
        if (sCnt <= i) return 0;
        if (visited[i]) return 0;
        if (!st[i] && !visited[i - 1]) return 0;
        
        visited[i] = 1;
    }
    
    int ret = 1;
    for (int i = 0; i < sCnt; i++)
        ret &= visited[i];
    return ret;
}

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	input();
	
	if (f()) cout << "Possible";
	else cout << "Impossible";
	
	return 0;
}
