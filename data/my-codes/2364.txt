#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
#define vi vector<int>
#define vl vector<ll>
#define vvi vector<vector<int>>
#define vvl vector<vector<ll>>

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;

	while (tc--) {
		int R, C;
		cin >> R >> C;
		
		vvi arr(R, vi(C));
		for (int y = 0; y < R; y++) {
			for (int x = 0; x < C; x++) {
				cin >> arr[y][x];
			}
		}

		int cntR = 0, cntC = 0;
		for (int y = 0; y < R; y++) {
			for (int x = 0; x < C; x++) {
				if (arr[y][x] == 1) break;
				if (x == C - 1) cntR++;
			}
		}
		for (int x = 0; x < C; x++) {
			for (int y = 0; y < R; y++) {
				if (arr[y][x] == 1) break;
				if (y == R - 1) cntC++;
			}
		}

		int cnt = min(cntR, cntC);

		if (cnt & 1) cout << "Ashish\n";
		else cout << "Vivek\n";
	}

	return 0;
}