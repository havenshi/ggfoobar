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

    def knockdown(self, pos):
        """Knock down a wall.
        Set the corresponding vertex to usable.
        Add edges from this vertex to adjacent vertices.
        Add edges from adjacent vertices to this vertex.
        :param pos: Walls position within maze
        :type pos: Tuple(x, y)
        """
        if self._knocked is not None:
            raise Exception("Cannot knock down more than one wall.")
        self._knocked = self.vertex[pos]
        self._knocked.usable = True
        for adj in self.get_adjacent(self.removed):
            self._knocked.edges.add(adj)
            adj.edges.add(self._knocked)

    def repair(self):
        """Repair a knocked down wall.
        If a wall has been knocked down, set the corresponding vertex
        to be unusable and remove all edges to and from this vertex.
        """
        if self.removed is not None:
            for v in self.removed.edges:
                v.edges.remove(self.removed)
            self.removed.edges = set()
            self.removed.usable = False
            self.removed = None

    def shortestpath(self):
        """Find the shortest path.
        Try to find shortest path with all walls intact.
        Then try to find the shortest path knocking each
        all down one at a time.
        :returns: Shortest path or None for no solution.
        :rtype: int|None
        """
        # Try with original walls:
        min_d, min_path = self._shortest_path()
        if min_d == self.distance:
            return min_d

        # Brute force removing at most 1 wall
        for wall in self.walls:
            self.knockdown(wall)
            try:
                d, path = self._shortest_path()
                if d < min_d:
                    min_d = d
                    min_path = path
                    if min_d == self.distance:
                        break
            finally:
                self.repair()
        return min_d

    def _shortest_path(self):
        """Find the shortest path with the current state of the maze.
        Maximum maze size is 20x20. The shortest path will be less than
        400. Default distances to 500. If the distance to a node is 500
        then there is no path to that node.
        :returns: Tuple of (distance, list of positions)
        :rtype: tuple
        """
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

        if r < 500:
            current = self.stop
            path = []
            while current is not None:
                path.append(current)
                current = prev[current.pos]
            return r + 1, list(reversed(path))
        return (500, None)

    def __str__(self):
        return "\n".join([str(r) for r in self.maze])


def answer(maze):
    return Maze(maze).shortestpath()

maze =  [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]
arr = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]

print answer(arr)