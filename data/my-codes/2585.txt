#include <iostream>
#include <algorithm>
#include <vector>
#define ll long long
using namespace std;

string s;
bool first[1000], second[1000];
vector<int> ans[1000];
int top;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	cin >> s;

	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '(') first[i] = true;
		else second[i] = true;
	}

	while (true) {
		int L = 0, R = s.size() - 1;
		while (true) {
			while (!first[L]) L++;
			if (!(L < R)) break;
			while (!second[R]) R--;
			if (!(L < R)) break;

			first[L] = false;
			second[R] = false;

			ans[top].push_back(L + 1);
			ans[top].push_back(R + 1);
		}

		if (ans[top].empty()) break;

		top++;
	}

	cout << top << '\n';
	for (int i = 0; i < top; i++) {
		sort(ans[i].begin(), ans[i].end());

		cout << ans[i].size() << '\n';
		for (int l = 0; l < ans[i].size(); l++)
			cout << ans[i][l] << ' ';
		cout << '\n';
	}

	return 0; 
}