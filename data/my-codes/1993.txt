#include <iostream>
#include <string>
#include <vector>
using namespace std;

vector<int> getpi(const string& s);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	while (true) {
		string s;
		getline(cin, s);

		if (s == ".") break;

		vector<int> pi = getpi(s);

		int ans = 1;
		for (int n = 2; n <= s.size(); n++) {
			if (s.size() % n) continue;
			if (pi[s.size() - 1] == s.size() * (n - 1) / n) ans = n;
		}

		cout << ans << '\n';
	}

	return 0;
}

vector<int> getpi(const string& s) {
	vector<int> pi(s.size());

	int begin = 1, matched = 0;
	while (begin + matched < s.size()) {
		if (s[begin + matched] == s[matched]) {
			matched++;
			pi[begin + matched - 1] = matched;
		}
		else {
			if (matched == 0) begin++;
			else {
				begin += matched - pi[matched - 1];
				matched = pi[matched - 1];
			}
		}
	}

	return pi;
}