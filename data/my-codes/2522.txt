#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N; cin >> N;

		vector<int> cnt(N + 1);
		map<int, int> mp;

		for (int i = 0; i < N; i++) {
			int x; cin >> x;
			cnt[x]++;
			mp.insert({ x, x });
		}

		int mxCnt = -1;
		for (int i = 1; i <= N; i++)
			mxCnt = max(mxCnt, cnt[i]);

		cout << max(min(mxCnt, (int)mp.size() - 1), min(mxCnt - 1, (int)mp.size())) << endl;
	}

	return 0;
}