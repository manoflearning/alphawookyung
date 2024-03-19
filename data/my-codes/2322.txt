#include <bits/stdc++.h>
#define ll long long
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

const int INF = 1e9 + 7;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int T; cin >> T;

	while (T--) {
		int N, K;
		cin >> N >> K;

		vector<int> cnt(26);
		for (int i = 0; i < N; i++) {
			char x; cin >> x;
			cnt[x - 'a']++;
		}
		
		vector<string> ans;
		int cK = K;
		for (int i = 0; i < 26; i++) {
			if (cK == 0) break;
			while (cnt[i] > 0) {
				string s(1, i + 'a');
				
				ans.push_back(s);
				cnt[i]--, cK--;
				if (cK == 0) break;
			}
		}
		
		if (ans[0] == ans[K - 1]) {
			int many = 0;
			for (int i = 0; i < 26; i++) {
				if (cnt[i] > 0) many++;
			}

			if (many > 1) {
				for (int i = 0; i < 26; i++) {
					while (cnt[i] > 0) {
						ans[0].push_back(i + 'a');
						cnt[i]--;
					}
				}
			}
			else {
				for (int i = 0; i < 26; i++) {
					while (cnt[i] > 0) {
						for (int l = 0; l < K; l++) {
							ans[l].push_back(i + 'a');
							cnt[i]--;
							if (cnt[i] == 0) break;
						}
					}
				}
			}
			
			cout << ans[0] << '\n';
		}
		else cout << ans[K - 1] << '\n';
	}

	return 0;
}