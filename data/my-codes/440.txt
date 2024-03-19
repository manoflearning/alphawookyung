#include <iostream>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int arr[1005];

	int loc = 1;
	for (int i = 1; loc <= 1000; i++) {
		for (int l = 1; l <= i; l++) {
			arr[loc++] = i;
			if (loc > 1000) break;
		}
	}

	int A, B;
	cin >> A >> B;

	int ans = 0;
	for (int i = A; i <= B; i++) {
		ans += arr[i];
	}

	cout << ans << '\n';

	return 0;
}