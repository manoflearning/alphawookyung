#include <iostream>
using namespace std;

const int dy[] = { 0, 0, 1, -1 };
const int dx[] = { 1, -1, 0, 0 };

int N, per[4];
bool visited[29][29];

double f(int y, int x, int cnt);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	cout << fixed; cout.precision(9);
	
	cin >> N;
	for (int& i : per) cin >> i;

	cout << f(14, 14, 0);

	return 0;
}

double f(int y, int x, int cnt) {
	if (visited[y][x]) return 0;
	if (cnt == N) return 1;

	double ret = 0;
	
	visited[y][x] = true;
	for (int i = 0; i < 4; i++) {
		int ny = y + dy[i], nx = x + dx[i];
		ret += f(ny, nx, cnt + 1) * per[i] / 100;
	}
	visited[y][x] = false;

	return ret;
}