#include <iostream>
#include <vector>
using namespace std;
#define pii pair<int, int>

const string dir = "UDLR";
const int dy[] = { -1, 1, 0, 0 };
const int dx[] = { 0, 0, -1, 1 };

int R, C;
string arr[1005];
vector<pii> adj[1005][1005];
bool visited[1005][1005];

void dfs(int y, int x);

int main() {
	cin.tie(NULL); cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	
	cin >> R >> C;

	for (int i = 0; i < R; i++) {
		cin >> arr[i];
	}

	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			int flag = dir.find(arr[y][x]);
			int ny = y + dy[flag], nx = x + dx[flag];
			
			adj[y][x].push_back({ ny, nx });
			adj[ny][nx].push_back({ y, x });
		}
	}

	int ans = 0;
	for (int y = 0; y < R; y++) {
		for (int x = 0; x < C; x++) {
			if (!visited[y][x]) {
				dfs(y, x);
				ans++;
			}
		}
	}

	cout << ans;

	return 0;
}

void dfs(int y, int x) {
	visited[y][x] = true;
	
	for (auto& i : adj[y][x]) {
		if (!visited[i.first][i.second])
			dfs(i.first, i.second);
	}
}