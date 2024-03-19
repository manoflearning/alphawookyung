#include <iostream>
#include <vector>
using namespace std;

vector<vector<char>> arr;

void f(int N, int y, int x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);

	int N; cin >> N;

	arr.resize(N, vector<char>(2 * N - 1, ' '));

	f(N, 0, N - 1);

	for (int y = 0; y < N; y++) {
		for (int x = 0; x < 2 * N - 1; x++)
			cout << arr[y][x];
		cout << '\n';
	}

	return 0;
}

void f(int N, int y, int x) {
	if (N == 3) {
		arr[y][x] = '*';
		arr[y + 1][x - 1] = arr[y + 1][x + 1] = '*';
		for (int i = -2; i <= 2; i++)
			arr[y + 2][x + i] = '*';
		return;
	}
	
	f(N / 2, y, x);
	f(N / 2, y + N / 2, x - N / 2);
	f(N / 2, y + N / 2, x + N / 2);
}