#include <iostream>
#include <set>
#include <vector>
using namespace std;

int N;
vector<bool> mAlpa(26, true), mWord((int)1e4 + 5, true);
vector<int> adj[26];
set<char> word[(int)1e4 + 5];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int M;
	cin >> N >> M;

	for (int i = 1; i <= N; i++) {
		string s; cin >> s;
		for (auto x : s) word[i].insert(x);
		for (auto x : word[i]) adj[x - 'a'].push_back(i);
	}

	int cnt = N;
	
	while (M--) {
		int q; char x;
		cin >> q >> x;
		
		if (q == 1) {
			mAlpa[x - 'a'] = false;
			for (int w : adj[x - 'a']) {
				if (mWord[w]) {
					mWord[w] = false;
					cnt--;
				}
			}
		}
		else {
			mAlpa[x - 'a'] = true;
			for (int w : adj[x - 'a']) {
				bool mCom = true;
				for (auto a : word[w]) {
					if (!mAlpa[a - 'a']) {
						mCom = false;
						break;
					}
				}

				if (mCom) {
					mWord[w] = true;
					cnt++;
				}
			}
		}

		cout << cnt << '\n';
	}

	return 0;
}