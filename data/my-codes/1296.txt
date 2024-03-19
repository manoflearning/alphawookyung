#include <iostream>
#include <vector>
#define ll long long
using namespace std;

const int MOD = 1e3;

int N;
vector<vector<int>> arr;

vector<vector<int>> f(ll x);
vector<vector<int>> matrixMul(vector<vector<int>>& a, vector<vector<int>>& b);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	ll B;
	cin >> N >> B;

	arr.resize(N, vector<int>(N));
	for (auto& i : arr)
		for (auto& l : i) cin >> l;

	vector<vector<int>> ans = f(B);
	for (auto& i : ans) {
		for (auto& l : i) cout << l % MOD << ' ';
		cout << '\n';
	}

	return 0;
}

vector<vector<int>> f(ll x) {
	if (x == 1) return arr;
	vector<vector<int>> prv = f(x >> 1);
	vector<vector<int>> ret = matrixMul(prv, prv);
	if (x % 2) ret = matrixMul(ret, arr);
	return ret;
}

vector<vector<int>> matrixMul(vector<vector<int>>& a, vector<vector<int>>& b) {
	vector<vector<int>> ret(N, vector<int>(N));
	for (int i = 0; i < N; i++) {
		for (int l = 0; l < N; l++) {
			for (int k = 0; k < N; k++) {
				ret[i][l] = (ret[i][l] + a[i][k] * b[k][l]) % MOD;
			}
		}
	}
	return ret;
}