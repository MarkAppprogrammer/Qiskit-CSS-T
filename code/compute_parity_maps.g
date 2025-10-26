LoadPackage("guava","0");;

G1:=[
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
];

G2:=[
[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
[1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0],
[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
];

gens := [G1, G2];
Hs := [];

for G in gens do
    GC := G * Z(2);
    code := GeneratorMatCode(GC, GF(2));
    Add(Hs, CheckMat(code));
od;

filepath := "/Users/mark/advocate/Qiskit-CSS-T/matrices/parity_maps.gap";
f := OutputTextFile(filepath, false);
PrintTo(f, "Hs := ", Hs, ";\n");
CloseStream(f);

Print("Stored H1 and H2 into parity_mats.gap\n");
