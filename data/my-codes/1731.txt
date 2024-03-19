#include <iostream>
using namespace std;

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	if (N & 1) {
		for (int i = 0; i < N; i++) cout << '*';
		cout << '\n';
		for (int i = N / 2 + 1; i >= 1; i--) {
			for (int l = 1; l <= N; l++) {
				if (N - l + 1 < i) break;
				if (l == i || N - l + 1 == i) cout << '*';
				else cout << ' ';
			}
			cout << '\n';
		}
	}
	else cout << "I LOVE CBNU\n";

	return 0;
}