#include <iostream>
#include <string>
using namespace std;

char trans(char x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	string s;

	getline(cin, s);

	for (int i = 0; i < s.size(); i++) {
		cout << trans(s[i]);
	}

	return 0;
}

char trans(char x) {
	if ('a' <= x && x <= 'z') {
		if (x + 13 <= 'z') return x + 13;
		else return x + 13 - ('z' - 'a' + 1);
	}
	else if ('A' <= x && x <= 'Z') {
		if (x + 13 <= 'Z') return x + 13;
		else return x + 13 - ('Z' - 'A' + 1);
	}
	else return x;
}