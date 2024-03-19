#include <iostream>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int tc; cin >> tc;
    int DSFDF;
	while (tc--) {
		int x, y, n; 
		cin >> x >> y >> n;

		int s = n - y;
		cout << s - s % x + y << '\n';
	}

	return 0;
}