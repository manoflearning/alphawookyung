#include <iostream>
#include <vector>
#include <algorithm>
#define pii pair<int, int>
using namespace std;

int N;
vector<pii> arr;
vector<vector<int>> dp(500, vector<int>(500, -1));

int find_min(int L, int R);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> N;

	arr.resize(N);
	for (auto& i : arr) 
		cin >> i.first >> i.second;

	cout << find_min(0, N - 1);

	return 0;
}

int find_min(int L, int R) {
	if (L == R) return 0;

	int& ret = dp[L][R];
	if (ret != -1) return ret;
	
	ret = (1 << 31) - 1;

	for (int i = L; i <= R - 1; i++) {
		ret = min(ret, arr[L].first * arr[i].second * arr[R].second + find_min(L, i) + find_min(i + 1, R));
	}

	return ret;
}