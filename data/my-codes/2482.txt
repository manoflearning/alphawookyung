#include <bits/stdc++.h>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int N, K;
		cin >> N >> K;

		string s; cin >> s;

		vector<vector<int>> cnt(K, vector<int>(26));

		for (int i = 0; i < s.size(); i++) {
			cnt[i % K][s[i] - 'a']++;
		}

		vector<char> alpha(K);

		for (int i = 0; i < K / 2 + 1; i++) {
			int flag = -1;
			char a;
			for (int l = 'a'; l <= 'z'; l++) {
				if (flag < cnt[i][l - 'a'] + cnt[K - i - 1][l - 'a']) {
					flag = cnt[i][l - 'a'] + cnt[K - i - 1][l - 'a'];
					a = (char)l;
				}
			}

			alpha[i] = alpha[K - i - 1] = a;
		}

		int ans = 0;
		for (int i = 0; i < s.size(); i++) {
			if (s[i] != alpha[i % K]) ans++;
		}

		cout << ans << '\n';
	}

	return 0;
}