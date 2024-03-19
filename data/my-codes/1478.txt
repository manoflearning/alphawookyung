#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

string ans[(int)1e4];

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	string s;
	getline(cin, s);

	int flag = 0;

	string first;
	for (; flag < s.size(); flag++) {
		if (s[flag] == ' ') {
			flag++;
			break;
		}

		first.push_back(s[flag]);
	}

	for (int i = 0; flag < s.size(); i++) {
		string x;
		for (; flag < s.size(); flag++) {
			if (('a' <= s[flag] && s[flag] <= 'z') || ('A' <= s[flag] && s[flag] <= 'Z')) {
				x.push_back(s[flag]);
			}
			else break;
		}

		string mid;
		for (; flag < s.size(); flag++) {
			if (s[flag] == ',' || s[flag] == ';') {
				flag += 2;
				break;
			}

			mid.push_back(s[flag]);
		}
		for (int l = 0; l < mid.size(); l++) {
			if (mid[l] == '[') {
				swap(mid[l], mid[l + 1]);
				l++;
			}
		}

		for (char c : first) ans[i].push_back(c);
		for (int l = mid.size() - 1; l >= 0; l--) {
			ans[i].push_back(mid[l]);
		}
		ans[i].push_back(' ');
		for(char c : x) ans[i].push_back(c);
		ans[i].push_back(';');
	}
	
	for (int i = 0; !ans[i].empty(); i++) cout << ans[i] << '\n';

	return 0;
}