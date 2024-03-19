#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>

const int INF = 1e9 + 7;
const int MOD = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };
const int MAX = 1e4;

vector<int> sX[2 * MAX + 5], sY[2 * MAX + 5];
vector<ll> psumX[2 * MAX + 5], rpsumX[2 * MAX + 5], psumY[2 * MAX + 5], rpsumY[2 * MAX + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int N; cin >> N;
	for (int i = 0; i < N; i++) {
		int x, y;
		cin >> x >> y;
		sX[x + MAX].push_back(y + MAX);
		sY[y + MAX].push_back(x + MAX);
	}

	for (int i = 0; i <= 2 * MAX; i++) {
		sort(sX[i].begin(), sX[i].end());
		sort(sY[i].begin(), sY[i].end());
		
		psumX[i].resize(sX[i].size());
		rpsumX[i].resize(sX[i].size());
		psumY[i].resize(sY[i].size());
		rpsumY[i].resize(sY[i].size());
		
		for (int j = 1; j < sX[i].size(); j++)
			psumX[i][j] = psumX[i][j - 1] + j * (sX[i][j] - sX[i][j - 1]);
		for (int j = sX[i].size() - 2; j >= 0; j--)
			rpsumX[i][j] = rpsumX[i][j + 1] + (sX[i].size() - (j + 1)) * (sX[i][j + 1] - sX[i][j]);
		for (int j = 1; j < sY[i].size(); j++)
			psumY[i][j] = psumY[i][j - 1] + j * (sY[i][j] - sY[i][j - 1]);
		for (int j = sY[i].size() - 2; j >= 0; j--)
			rpsumY[i][j] = rpsumY[i][j + 1] + (sY[i].size() - (j + 1)) * (sY[i][j + 1] - sY[i][j]);
	}

	ll res = 0;
	for (int i = 0; i <= 2 * MAX; i++) {
		for (int j = 0; j < sX[i].size(); j++) {
			int k = lower_bound(sY[sX[i][j]].begin(), sY[sX[i][j]].end(), i) - sY[sX[i][j]].begin();
			res += (psumX[i][j] + rpsumX[i][j]) * (psumY[sX[i][j]][k] + rpsumY[sX[i][j]][k]);
			res %= MOD;
		}
	}

	cout << res;

	return 0;
}