#include <bits/stdc++.h>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int T; cin >> T;

	while (T--) {
		int Y, X;
		cin >> Y >> X;

		if (Y % 2 && X % 2) {
			for (int y = 1; y <= Y; y++) {
				for (int x = 1; x <= X; x++) {
					if ((y % 2 && x % 2) || (!(y % 2) && !(x % 2))) cout << "B";
					else cout << "W";
				}
				cout << endl;
			}
		}
		else if (Y % 2 || X % 2) {
			for (int y = 1; y <= Y - 1; y++) {
				for (int x = 1; x <= X; x++) {
					if ((y % 2 && x % 2) || (!(y % 2) && !(x % 2))) cout << "B";
					else cout << "W";
				}
				cout << endl;
			}
			for (int y = Y; y <= Y; y++) {
				for (int x = 1; x <= X - 1; x++) {
					if ((y % 2 && x % 2) || (!(y % 2) && !(x % 2))) cout << "B";
					else cout << "W";
				}
				cout << "B" << endl;
			}
		}
		else {
			for (int y = 1; y <= Y - 1; y++) {
				for (int x = 1; x <= X; x++) {
					if ((y % 2 && x % 2) || (!(y % 2) && !(x % 2))) cout << "B";
					else cout << "W";
				}
				cout << endl;
			}
			for (int y = Y; y <= Y; y++) {
				for (int x = 1; x <= X - 2; x++) {
					if ((y % 2 && x % 2) || (!(y % 2) && !(x % 2))) cout << "B";
					else cout << "W";
				}
				cout << "BB" << endl;
			}
		}
	}

	return 0;
}