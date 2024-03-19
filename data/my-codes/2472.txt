#include <iostream>
using namespace std;

int main() {
	int t;
	cin >> t;

	for (int test = 0; test < t; test++) {
		double x;
		cin >> x;

		double ans = x;
		for (int i = 1; i <= 10; i++) {
			x /= 10;
			ans += x;
		}

		cout << (int)ans << '\n';
	}

	return 0;
}