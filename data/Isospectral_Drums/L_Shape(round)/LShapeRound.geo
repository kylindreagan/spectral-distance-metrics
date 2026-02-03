cl__1 = 1;
Point(1) = {0, 1, 0, 1};
Point(2) = {-1, 1, 0, 1};
Point(3) = {-1, -1, 0, 1};
Point(4) = {1, -1, 0, 1};
Point(5) = {1, 0, 0, 1};
Point(6) = {1, 1, 0, 1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Circle(5) = {5, 6, 1};
Delete {
  Point{6};
}
Line Loop(6) = {1, 2, 3, 4, 5};
Plane Surface(7) = {6};
Point(7) = {0, 0.89, 0, 1.0};
Point(8) = {0.03, 0.75, 0, 1.0};
Point(9) = {0.07, 0.64, 0, 1.0};
Point(10) = {0.1, 0.55, 0, 1.0};
Point(11) = {0.16, 0.46, 0, 1.0};
Point(12) = {0.24, 0.35, 0, 1.0};
Point(13) = {0.32, 0.27, 0, 1.0};
Point(14) = {0.43, 0.18, 0, 1.0};
Point(15) = {0.55, 0.11, 0, 1.0};
Point(16) = {0.68, 0.05, 0, 1.0};
Point(17) = {0.78, 0.02, 0, 1.0};
Point(18) = {0.89, 0.01, 0, 1.0};
Plane Surface(8) = {6};
Ruled Surface(9) = {6};
Delete {
  Surface{9};
}
Delete {
  Surface{7};
}
Delete {
  Surface{8};
}
Plane Surface(8) = {6};
Point{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18} In Surface{8};
