import 'dart:async';

import 'dart:convert';

import 'dart:collection';

import 'dart:math';

import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'package:provider/provider.dart';

import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';

import 'package:permission_handler/permission_handler.dart';

// --- App Entry Point ---

void main() {
  runApp(const MazeApp());
}

class MazeApp extends StatelessWidget {
  const MazeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Maze Robot Controller',
      theme: ThemeData(
        primarySwatch: Colors.indigo,
        scaffoldBackgroundColor: Colors.grey[200],
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.indigo,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          ),
        ),
      ),
      home: ChangeNotifierProvider(
        create: (_) => BluetoothMazeController(),
        child: const MazeHomePage(),
      ),
    );
  }
}

// --- Enums for Maze and Robot State ---

enum CellType { path, wall, start, end, solution }

enum Tool { wall, path, start, end }

enum Direction { up, down, left, right, none }

// --- Main App State Management ---

class BluetoothMazeController extends ChangeNotifier {
// Maze properties

  int rows = 100; // Maximum grid size

  int cols = 100; // Maximum grid size

  late List<List<CellType>> grid;

  Point<int>? startPoint;

  Point<int>? endPoint;

  Tool currentTool = Tool.path;

  List<Point<int>> solutionPath = [];

  String solutionString = '';

  bool isAnimatingSolution = false;

  Timer? _animationTimer;

// Bluetooth properties

  final FlutterBluetoothSerial _bluetooth = FlutterBluetoothSerial.instance;

  List<BluetoothDevice> _devicesList = [];

  BluetoothDevice? selectedDevice;

  BluetoothConnection? connection;

  String connectionStatus = "Disconnected";

  bool isConnecting = false;

// Robot command properties

  List<String> _commands = [];

  BluetoothMazeController() {
    _initializeGrid();
  }

  @override
  void dispose() {
    connection?.dispose();

    _animationTimer?.cancel();

    super.dispose();
  }

  void _initializeGrid() {
    grid =
        List.generate(rows, (_) => List.generate(cols, (_) => CellType.wall));

    startPoint = null;

    endPoint = null;

    clearSolution();
  }

// --- UI Logic ---

  void setTool(Tool tool) {
    currentTool = tool;

    notifyListeners();
  }

  void updateCell(int row, int col) {
    if (isAnimatingSolution) return;

    if (row < 0 || row >= rows || col < 0 || col >= cols) return;

    final point = Point(col, row);

    if (currentTool == Tool.start) {
      if (startPoint != null) {
        grid[startPoint!.y][startPoint!.x] = CellType.path;
      }

      startPoint = point;
    } else if (currentTool == Tool.end) {
      if (endPoint != null) {
        grid[endPoint!.y][endPoint!.x] = CellType.path;
      }

      endPoint = point;
    }

    switch (currentTool) {
      case Tool.wall:
        grid[row][col] = CellType.wall;

        break;

      case Tool.path:
        grid[row][col] = CellType.path;

        break;

      case Tool.start:
        grid[row][col] = CellType.start;

        break;

      case Tool.end:
        grid[row][col] = CellType.end;

        break;
    }

    clearSolution();

    notifyListeners();
  }

  void resetMaze() {
    _initializeGrid();

    notifyListeners();
  }

  void clearSolution() {
    _animationTimer?.cancel();

    isAnimatingSolution = false;

    for (var point in solutionPath) {
      if (point.y < grid.length &&
          point.x < grid[0].length &&
          grid[point.y][point.x] == CellType.solution) {
        grid[point.y][point.x] = CellType.path;
      }
    }

    solutionPath.clear();

    solutionString = '';

    notifyListeners();
  }

// --- Pathfinding Logic (BFS) ---

  void solveMaze() {
    if (startPoint == null || endPoint == null) {
      solutionString = "Please set a Start and End point.";

      notifyListeners();

      return;
    }

    clearSolution();

    Queue<List<Point<int>>> queue = Queue();

    Set<Point<int>> visited = {};

    queue.add([startPoint!]);

    visited.add(startPoint!);

    while (queue.isNotEmpty) {
      var path = queue.removeFirst();

      var lastPoint = path.last;

      if (lastPoint == endPoint) {
        solutionPath = path;

        _animateSolution();

        return;
      }

      var neighbors = [
        Point(lastPoint.x, lastPoint.y - 1),
        Point(lastPoint.x, lastPoint.y + 1),
        Point(lastPoint.x - 1, lastPoint.y),
        Point(lastPoint.x + 1, lastPoint.y),
      ];

      for (var neighbor in neighbors) {
        if (neighbor.y >= 0 &&
            neighbor.y < rows &&
            neighbor.x >= 0 &&
            neighbor.x < cols &&
            grid[neighbor.y][neighbor.x] != CellType.wall &&
            !visited.contains(neighbor)) {
          visited.add(neighbor);

          var newPath = List<Point<int>>.from(path);

          newPath.add(neighbor);

          queue.add(newPath);
        }
      }
    }

    solutionString = "No solution found!";

    notifyListeners();
  }

  void _animateSolution() {
    if (solutionPath.isEmpty) return;

    isAnimatingSolution = true;

    int animationIndex = 0;

    _animationTimer = Timer.periodic(const Duration(milliseconds: 10), (timer) {
      if (animationIndex < solutionPath.length) {
        final point = solutionPath[animationIndex];

        if (grid[point.y][point.x] == CellType.path) {
          grid[point.y][point.x] = CellType.solution;
        }

        animationIndex++;

        notifyListeners();
      } else {
        timer.cancel();

        isAnimatingSolution = false;

        _generateCommands();

        solutionString = _commands.join(' → ');

        notifyListeners();
      }
    });
  }

// --- Bluetooth Logic ---

  Future<void> getPairedDevices() async {
    await [Permission.bluetoothScan, Permission.bluetoothConnect].request();

    connectionStatus = "Getting paired devices...";

    notifyListeners();

    try {
      _devicesList = await _bluetooth.getBondedDevices();
    } catch (e) {
      connectionStatus = "Error: ${e.toString()}";
    }

    connectionStatus = "Select a device";

    notifyListeners();
  }

  List<BluetoothDevice> get devices => _devicesList;

  void connectToDevice(BluetoothDevice device) async {
    if (isConnecting || connection != null) return;

    isConnecting = true;

    selectedDevice = device;

    connectionStatus = "Connecting...";

    notifyListeners();

    try {
      connection = await BluetoothConnection.toAddress(device.address);

      connectionStatus = "Connected to ${device.name}";

      isConnecting = false;

      connection!.input!.listen(null).onDone(() => disconnectFromDevice());
    } catch (e) {
      connectionStatus = "Connection failed: ${e.toString()}";

      isConnecting = false;
    }

    notifyListeners();
  }

  void disconnectFromDevice() {
    connection?.dispose();

    connection = null;

    selectedDevice = null;

    connectionStatus = "Disconnected";

    notifyListeners();
  }

// --- Robot Control Logic ---

  void startRobot() async {
    if (solutionPath.isEmpty || connection == null) {
      connectionStatus = "Not ready. Solve & connect.";

      notifyListeners();

      return;
    }

    _generateCommands();

    String fullCommandString = "${_commands.join('')}\n";

    if (fullCommandString.trim().isEmpty) {
      connectionStatus = "No commands to send.";

      notifyListeners();

      return;
    }

    connectionStatus = "Sending commands...";

    notifyListeners();

    try {
      Uint8List bytes = Uint8List.fromList(utf8.encode(fullCommandString));

      connection!.output.add(bytes);

      await connection!.output.allSent;

      connectionStatus = "Commands sent!";
    } catch (e) {
      connectionStatus = "Send failed: ${e.toString()}";
    }

    notifyListeners();
  }

  void _generateCommands() {
    _commands.clear();

    if (solutionPath.length < 2) return;

    List<int> decisionPointIndices = [0];

    for (int i = 1; i < solutionPath.length - 1; i++) {
      Point<int> current = solutionPath[i];

      Direction dirIn = _getDirection(solutionPath[i - 1], current);

      Direction dirOut = _getDirection(current, solutionPath[i + 1]);

      bool isCorner = (dirIn != dirOut);

      bool isIntersection = false;

      if (!isCorner) {
        if (dirIn == Direction.up || dirIn == Direction.down) {
          if ((current.x > 0 &&
                  grid[current.y][current.x - 1] != CellType.wall) ||
              (current.x < cols - 1 &&
                  grid[current.y][current.x + 1] != CellType.wall)) {
            isIntersection = true;
          }
        } else {
          if ((current.y > 0 &&
                  grid[current.y - 1][current.x] != CellType.wall) ||
              (current.y < rows - 1 &&
                  grid[current.y + 1][current.x] != CellType.wall)) {
            isIntersection = true;
          }
        }
      }

      if (isCorner || isIntersection) {
        decisionPointIndices.add(i);
      }
    }

    Direction currentDirection =
        _getDirection(solutionPath[0], solutionPath[1]);

    for (int j = 0; j < decisionPointIndices.length; j++) {
      int index = decisionPointIndices[j];

      if (index >= solutionPath.length - 1) break;

      Direction requiredDirection =
          _getDirection(solutionPath[index], solutionPath[index + 1]);

      if (j > 0) {
        _commands.addAll(_getTurnCommands(currentDirection, requiredDirection));
      }

      _commands.add('F');

      currentDirection = requiredDirection;
    }
  }

  Direction _getDirection(Point<int> from, Point<int> to) {
    if (to.y > from.y) return Direction.down;

    if (to.y < from.y) return Direction.up;

    if (to.x > from.x) return Direction.right;

    if (to.x < from.x) return Direction.left;

    return Direction.none;
  }

  List<String> _getTurnCommands(Direction current, Direction required) {
    if (current == required) return [];

    if ((current == Direction.down && required == Direction.left) ||
        (current == Direction.up && required == Direction.right) ||
        (current == Direction.left && required == Direction.up) ||
        (current == Direction.right && required == Direction.down))
      return ['R'];

    if ((current == Direction.down && required == Direction.right) ||
        (current == Direction.up && required == Direction.left) ||
        (current == Direction.left && required == Direction.down) ||
        (current == Direction.right && required == Direction.up)) return ['L'];

    return ['R', 'R'];
  }
}

// --- UI Widgets ---

class MazeHomePage extends StatelessWidget {
  const MazeHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar:
          AppBar(title: const Text('Maze Robot Controller'), centerTitle: true),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Column(
            children: [
              const BluetoothControlPanel(),
              const SizedBox(height: 16),
              const MazeGrid(),
              const SizedBox(height: 16),
              Consumer<BluetoothMazeController>(
                builder: (context, model, child) {
                  if (model.solutionString.isEmpty &&
                      !model.isAnimatingSolution) {
                    return const SizedBox.shrink();
                  }

                  return Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.withOpacity(0.3),
                          spreadRadius: 2,
                          blurRadius: 5,
                          offset: const Offset(0, 3),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        const Text("Solution Path",
                            style: TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 16)),
                        const SizedBox(height: 8),
                        if (model.isAnimatingSolution)
                          const CircularProgressIndicator()
                        else
                          Text(
                            model.solutionString,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                                fontSize: 14, color: Colors.indigo),
                          ),
                      ],
                    ),
                  );
                },
              ),
              const SizedBox(height: 16),
              const ToolSelector(),
              const SizedBox(height: 16),
              const MazeControls(),
            ],
          ),
        ),
      ),
    );
  }
}

class BluetoothControlPanel extends StatelessWidget {
  const BluetoothControlPanel({super.key});

  @override
  Widget build(BuildContext context) {
    final model = Provider.of<BluetoothMazeController>(context);

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            Text(model.connectionStatus,
                style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.search),
                  label: const Text('Find Devices'),
                  onPressed: model.isConnecting ? null : model.getPairedDevices,
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.cancel),
                  label: const Text('Disconnect'),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                  onPressed: model.connection == null
                      ? null
                      : model.disconnectFromDevice,
                ),
              ],
            ),
            if (model.devices.isNotEmpty && model.connection == null)
              SizedBox(
                height: 120,
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: model.devices.length,
                  itemBuilder: (context, index) {
                    final device = model.devices[index];

                    return ListTile(
                      title: Text(device.name ?? "Unknown"),
                      subtitle: Text(device.address),
                      onTap: () => model.connectToDevice(device),
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ** FINAL, ROBUST INTERACTIVE MAZE GRID WIDGET **

class MazeGrid extends StatefulWidget {
  const MazeGrid({super.key});

  @override
  State<MazeGrid> createState() => _MazeGridState();
}

class _MazeGridState extends State<MazeGrid> {
  final TransformationController _transformationController =
      TransformationController();

  final double _cellSize = 20.0;

  bool _hasBeenInitialized = false;

  @override
  void dispose() {
    _transformationController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final model = Provider.of<BluetoothMazeController>(context);

    final totalWidth = model.cols * _cellSize;

    final totalHeight = model.rows * _cellSize;

    return Card(
      clipBehavior: Clip.antiAlias,
      elevation: 4,
      child: AspectRatio(
        aspectRatio: 1.0,
        child: LayoutBuilder(
          builder: (context, constraints) {
            double minScale = 0.1;

            if (constraints.maxWidth > 0 && totalWidth > 0) {
              minScale = constraints.maxWidth / totalWidth;
            }

            if (!_hasBeenInitialized && minScale > 0.1) {
// Defer this call to avoid conflicts during build phase

              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (mounted) {
                  _transformationController.value = Matrix4.identity()
                    ..scale(minScale);

                  _hasBeenInitialized = true;
                }
              });
            }

            return InteractiveViewer(
              transformationController: _transformationController,

              boundaryMargin: const EdgeInsets.all(20.0),

              minScale: minScale, // Set dynamically

              maxScale: 5.0,

              child: GestureDetector(
                onTapUp: (details) {
                  final RenderBox box = context.findRenderObject() as RenderBox;

                  final Offset localPosition =
                      box.globalToLocal(details.globalPosition);

                  final Matrix4 inverseMatrix =
                      Matrix4.inverted(_transformationController.value);

                  final Offset transformedPosition =
                      MatrixUtils.transformPoint(inverseMatrix, localPosition);

                  final int col = (transformedPosition.dx / _cellSize).floor();

                  final int row = (transformedPosition.dy / _cellSize).floor();

                  model.updateCell(row, col);
                },
                child: CustomPaint(
                  size: Size(totalWidth, totalHeight),
                  painter: _MazePainter(
                    grid: model.grid,
                    cellSize: _cellSize,
                    scale: _transformationController.value.getMaxScaleOnAxis(),
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _MazePainter extends CustomPainter {
  final List<List<CellType>> grid;

  final double cellSize;

  final double scale;

  _MazePainter(
      {required this.grid, required this.cellSize, required this.scale});

  @override
  void paint(Canvas canvas, Size size) {
    final wallPaint = Paint()..color = Colors.white;

    final pathPaint = Paint()..color = Colors.black87;

    final startPaint = Paint()..color = Colors.green;

    final endPaint = Paint()..color = Colors.red;

    final solutionPaint = Paint()..color = Colors.tealAccent;

    final gridPaint = Paint()
      ..color = Colors.grey.shade300
      ..strokeWidth = 0.5 /
          (scale < 0.1 ? 0.1 : scale); // Prevent stroke from becoming too large

    for (int row = 0; row < grid.length; row++) {
      for (int col = 0; col < grid[0].length; col++) {
        final rect =
            Rect.fromLTWH(col * cellSize, row * cellSize, cellSize, cellSize);

        Paint currentPaint;

        switch (grid[row][col]) {
          case CellType.path:
            currentPaint = pathPaint;

            break;

          case CellType.wall:
            currentPaint = wallPaint;

            break;

          case CellType.start:
            currentPaint = startPaint;

            break;

          case CellType.end:
            currentPaint = endPaint;

            break;

          case CellType.solution:
            currentPaint = solutionPaint;

            break;
        }

        canvas.drawRect(rect, currentPaint);
      }
    }

    for (int i = 0; i <= grid.length; i++) {
      canvas.drawLine(
          Offset(0, i * cellSize), Offset(size.width, i * cellSize), gridPaint);
    }

    for (int i = 0; i <= grid[0].length; i++) {
      canvas.drawLine(Offset(i * cellSize, 0),
          Offset(i * cellSize, size.height), gridPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _MazePainter oldDelegate) {
    return true; // Keep as true to ensure instant UI updates
  }
}

class ToolSelector extends StatelessWidget {
  const ToolSelector({super.key});

  @override
  Widget build(BuildContext context) {
    final model = Provider.of<BluetoothMazeController>(context);

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      alignment: WrapAlignment.center,
      children: [
        _buildToolButton(context, model, Tool.path, Icons.edit_road, "Path"),
        _buildToolButton(context, model, Tool.wall, Icons.select_all, "Wall"),
        _buildToolButton(context, model, Tool.start, Icons.flag, "Start"),
        _buildToolButton(
            context, model, Tool.end, Icons.assistant_photo, "End"),
      ],
    );
  }

  Widget _buildToolButton(BuildContext context, BluetoothMazeController model,
      Tool tool, IconData icon, String label) {
    final isSelected = model.currentTool == tool;

    return ElevatedButton.icon(
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: isSelected ? Colors.amber[700] : Colors.grey[400],
        foregroundColor: isSelected ? Colors.white : Colors.black87,
      ),
      onPressed: () => model.setTool(tool),
    );
  }
}

class MazeControls extends StatelessWidget {
  const MazeControls({super.key});

  @override
  Widget build(BuildContext context) {
    final model = Provider.of<BluetoothMazeController>(context);

    final canStart = model.solutionPath.isNotEmpty &&
        model.connection != null &&
        !model.isAnimatingSolution;

    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton.icon(
              icon: const Icon(Icons.lightbulb_outline),
              label: const Text("Solve"),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.orange),
              onPressed: model.isAnimatingSolution ? null : model.solveMaze,
            ),
            ElevatedButton.icon(
              icon: const Icon(Icons.refresh),
              label: const Text("Reset"),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.blueGrey),
              onPressed: model.isAnimatingSolution ? null : model.resetMaze,
            ),
          ],
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            icon: const Icon(Icons.send),
            label: const Text("Start Robot"),
            style: ElevatedButton.styleFrom(
                backgroundColor: canStart ? Colors.green : Colors.grey,
                padding: const EdgeInsets.symmetric(vertical: 16)),
            onPressed: canStart ? model.startRobot : null,
          ),
        ),
      ],
    );
  }
}
