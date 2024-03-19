#include <iostream>
#include <queue>
#include <cmath>
using namespace std;

int t;
vector<pair<int, int>> v;

void bfs(int startx, int starty, int endx, int endy)
{
	queue<pair<int, int>> q;
	q.push(make_pair(startx, starty));

	while (!q.empty())
	{
		int x = q.front().first;
		int y = q.front().second;
		q.pop();

		if (endx == x && endy == y)
		{
			cout << "happy\n";
			return;
		}

		for (auto i = v.begin(); i != v.end();)
		{
			int xl = abs(x - i->first);
			int yl = abs(y - i->second);
			if ((xl + yl) <= 1000)
			{
				q.push(make_pair(i->first, i->second));
				i = v.erase(i);
			}
			else
				i++;
		}
	}

	cout << "sad\n";
}

int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(NULL);
	std::cout.tie(NULL);
	
	cin >> t;
	for (size_t testcase = 0; testcase < t; testcase++)
	{
		v.clear();
		int n, startx, starty, endx, endy;
		cin >> n;
		cin >> startx >> starty;
		for (size_t i = 0; i < n; i++)
		{
			int x, y;
			cin >> x >> y;
			v.push_back(make_pair(x, y));
		}
		cin >> endx >> endy;
		v.push_back(make_pair(endx, endy));
		bfs(startx, starty, endx, endy);
	}

	return 0;
}