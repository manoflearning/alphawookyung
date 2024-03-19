#include <iostream>
using namespace std;

int main() {
	int tc; cin >> tc;
	while (tc--) {
	    int n; cin >> n;
	    cout << n / 7 + (n % 7 >= 2 ? 1 : 0) << '\n';
	}
	return 0;
}
