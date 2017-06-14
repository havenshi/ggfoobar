class Vertex():
    # create vertex where the path starts from
    def __init__(self, pos, wall):
        self.edges = set()
        self.pos = pos
        self.usable = not wall # when wall = 0, it is usable; else, not usable.

    def __repr__(self):
        return "({}, {})".format(self.pos[0], self.pos[1]) # return its coordinate


class Maze():
    def __init__(self, maze):
        self.maze = maze
        self.height = len(maze)
        self.width = len(maze[0])
        self.distance = self.width + self.height - 1
        self.build_vertex()
        self.build_edges()
        self.removed = None

    def build_vertex(self):
        self.vertex = {}
        for j in range(len(self.maze)):
            for i in range(len(self.maze[j])):
                self.vertex[(i, j)] = Vertex((i, j), self.maze[j][i])
        self.start = self.vertex[(0, 0)]
        self.stop = self.vertex[(self.width - 1, self.height - 1)]

    def build_edges(self):
        # edges that is around vertex and do not has wall
        self.walls = []
        for each_v in self.vertex.itervalues():
            if not each_v.usable:
                self.walls.append(each_v.pos)
            for each_d in self.directions(each_v):
                each_v.edges.add(each_d)

    def directions(self, vertex):
        # four directions each vertex can go to
        x, y = vertex.pos
        directions = []
        if x - 1 >= 0 and self.vertex[(x - 1, y)].usable:
            directions.append(self.vertex[(x - 1, y)])

        if x + 1 < self.width and self.vertex[(x + 1, y)].usable:
            directions.append(self.vertex[(x + 1, y)])

        if y - 1 >= 0 and self.vertex[(x, y - 1)].usable:
            directions.append(self.vertex[(x, y - 1)])

        if y + 1 < self.height and self.vertex[(x, y + 1)].usable:
            directions.append(self.vertex[x, y + 1])
        return directions

    def remove(self, pos):
        self.removed = self.vertex[pos]
        self.removed.usable = True
        for adj in self.directions(self.removed):
            self.removed.edges.add(adj)
            adj.edges.add(self.removed)

    def recover(self):
        if self.removed is not None:
            for each_v in self.removed.edges:
                each_v.edges.remove(self.removed)
            self.removed.edges = set()
            self.removed.usable = False
            self.removed = None

    def shortestpath(self):
        # distance of original maza:
        min_d, min_path = self.helper()
        if min_d == self.distance:
            return min_d

        # distance of maze that removes at most one wall
        for wall in self.walls:
            self.remove(wall)
            d, path = self.helper()
            if d < min_d:
                min_d = d
                min_path = path
                if min_d == self.distance:
                    break
            self.recover() # return the wall back
        return min_d

    def helper(self):
        distance = {}
        prev = {}
        visited = set()
        for v in self.vertex.itervalues():
            distance[v.pos] = 500
            prev[v.pos] = None

        distance[self.start.pos] = 0
        q = [self.start]
        while len(q) > 0:
            current = q.pop(0)
            if current in visited:
                continue
            if current is self.stop and current in visited:
                break

            d = distance[current.pos] + 1
            for v in current.edges:
                if distance[v.pos] > d:
                    distance[v.pos] = d
                    prev[v.pos] = current
                q.append(v)
            visited.add(current)

        r = distance[self.stop.pos]
        current = self.stop
        path = []
        while current is not None:
            path.append(current)
            current = prev[current.pos]
        return r + 1, list(reversed(path))

    def __str__(self):
        return "\n".join([str(r) for r in self.maze])


def answer(maze):
    return Maze(maze).shortestpath()

maze =  [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]
arr = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]

print answer(arr)



# import collections
# def bfs(maze, start_point):
#     queue_positions = collections.deque()
#     queue_positions.append(start_point)
#     len_maze = len(maze[0])
#     hei_maze = len(maze)
#     step_map = [[0 for i in range(len_maze)] for j in range(hei_maze)]
#     step_map[start_point[0]][start_point[1]] = 1
#     while queue_positions:
#         c_pos = queue_positions.popleft()
#         if c_pos[1] == len_maze - 1 and c_pos[0] == hei_maze - 1:
#             return step_map
#         #go up
#         if c_pos[0] > 0 and step_map[c_pos[0] - 1][c_pos[1]] == 0 and maze[c_pos[0] - 1][c_pos[1]] == 0:
#             step_map[c_pos[0] - 1][c_pos[1]] = c_pos[2] + 1
#             queue_positions.append([c_pos[0] - 1, c_pos[1],c_pos[2] + 1])
#         #go down
#         if c_pos[0] < hei_maze - 1 and step_map[c_pos[0] + 1][c_pos[1]] == 0 and maze[c_pos[0] + 1][c_pos[1]] == 0:
#             step_map[c_pos[0] + 1][c_pos[1]] = c_pos[2] + 1
#             queue_positions.append([c_pos[0] + 1, c_pos[1],c_pos[2] + 1])
#         #go left
#         if c_pos[1] > 0 and step_map[c_pos[0]][c_pos[1] - 1] == 0 and maze[c_pos[0]][c_pos[1] - 1] == 0:
#             step_map[c_pos[0]][c_pos[1] - 1] = c_pos[2] + 1
#             queue_positions.append([c_pos[0], c_pos[1] - 1, c_pos[2] + 1])
#         #go right
#         if c_pos[1] < len_maze - 1 and step_map[c_pos[0]][c_pos[1] + 1] == 0 and maze[c_pos[0]][c_pos[1] + 1] == 0:
#             step_map[c_pos[0]][c_pos[1] + 1] = c_pos[2] + 1
#             queue_positions.append([c_pos[0], c_pos[1] + 1, c_pos[2] + 1])
#     return step_map
#
# # print bfs(maze, [0,0,1])
#
# def find_certified_pos(maze, step_map, pos):
#     count = 0
#     len_maze = len(maze[0])
#     hei_maze = len(maze)
#     least_step = 1 << 31
#     for i, j in [[1,0], [-1,0], [0,1], [0,-1]]:
#         i2 = pos[0] + i
#         j2 = pos[1] + j
#         if i2 < 0 or i2 == hei_maze or j2 < 0 or j2 == len_maze:
#             continue
#         elif maze[i2][j2] == 0:
#             count += 1
#             if step_map[i2][j2] > 0:
#                 least_step = min(step_map[i2][j2], least_step)
#     if count >= 2 or (pos[0] == hei_maze - 1 and pos[1] == len_maze - 1):
#         return least_step
#     else:
#         return False
#
# def answer(maze):
#     len_maze = len(maze[0])
#     hei_maze = len(maze)
#     step_map = bfs(maze, [0,0,1])
#     best_step = step_map[hei_maze-1][len_maze-1]
#     if best_step == len_maze + hei_maze - 1:
#         return best_step
#     least_step = 0
#     answer_step = 1 << 31
#     for i in range(hei_maze):
#         for j in range(len_maze):
#             if maze[i][j] == 1:
#                 least_step = find_certified_pos(maze, step_map, [i, j])
#             if least_step:
#                 step = bfs(maze, [i, j, 1])
#                 if step[hei_maze-1][len_maze-1] != 0:
#                     answer_step = min(least_step + step[hei_maze-1][len_maze-1], answer_step)
#             if answer_step == hei_maze + len_maze - 1:
#                 return answer_step
#     return answer_step
#
#
# maze =  [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]
# arr = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
# print answer(arr)



# import collections
#
# def answer(maze):
#     if not maze : return (1<<32)-1
#     m = len(maze)
#     if m == 0 : return (1<<32)-1
#     n = len(maze[0])
#     if n == 0 : return (1<<32)-1
#     visit = [[-1] * n for _ in range(m)]
#     dq = collections.deque()
#     dq.append((0, 0, 1, 1))
#     res = (1<<32)-1
#     while dq:
#         i, j, k, step = dq.popleft()
#         if i == m-1 and j == n-1 and step < res:
#             res = step
#             break
#         for a, b in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
#             ii = i + a
#             jj = j + b
#             kk = k
#             if ii < 0 or ii >= m or jj < 0 or jj >= n or (visit[ii][jj] != -1 and visit[ii][jj] < kk) or (maze[ii][jj] == 1 and kk < 0) : continue
#             if maze[ii][jj] == 1 : kk -= 1
#             visit[ii][jj] = kk
#             dq.append((ii, jj, kk, step+1))
#     return res