#include <iostream>
#include <string>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;
	cin.ignore();

	for (int i = 0; i < N; i++) {
		string s;
		getline(cin, s);

		string ans;
		
		cout << "god";
		bool start = true;
		for (int i = 0; i < s.size(); i++) {
			if (s[i] == ' ') start = false;
			else if (!start) cout << s[i];
		}
		cout << '\n';
	}
	
	return 0;
}