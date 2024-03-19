#include <iostream>
#include <cmath>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	int sum = 0, ans = 0;

	for (int i = 0; i < 10; i++) {
		int x; cin >> x;
		sum += x;
		if (fabs(100 - ans) >= fabs(100 - sum)) ans = sum;
	}

	cout << ans;

	return 0;
}