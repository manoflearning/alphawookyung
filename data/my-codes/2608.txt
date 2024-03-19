#include <iostream>
#include <algorithm>
#define ll long long
using namespace std;

string s;
ll size1[26], size2[26][26];

int main() {
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> s;

	for (int i = 0; i < s.size(); i++) {
		int flag = s[i] - 'a';
		for (int l = 0; l < 26; l++)
			size2[l][flag] += size1[l];

		size1[flag]++;
	}

	ll ans = 0;
	for (int i = 0; i < 26; i++) {
		for (int l = 0; l < 26; l++)
			ans = max(ans, size2[i][l]);
		ans = max(ans, size1[i]);
	}

	cout << ans;

	return 0;
}