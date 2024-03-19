#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;
const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N, M;
		cin >> N >> M;

		vector<string> arr(N);
		for (auto& i : arr) cin >> i;

		string ans = arr[0];
		bool YES = true;

		for (int x = 0; x < M; x++) {
			for (char chng = 'a'; chng <= 'z'; chng++) {
				ans[x] = chng;
				YES = true;

				for (int i = 1; i < N; i++) {
					int diffCnt = 0;
					for (int l = 0; l < M; l++) {
						if (ans[l] != arr[i][l]) diffCnt++;
					}
					
					if (diffCnt >= 2) YES = false;
				}
				
				if (YES) {
					x = M, chng = 'z' + 1;
					break;
				}
			}

			ans[x] = arr[0][x];
		}

		if (YES) cout << ans << '\n';
		else cout << "-1\n";
	}

	return 0;
}