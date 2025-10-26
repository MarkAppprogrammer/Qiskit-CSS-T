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

for i in [1..Length(gens)] do
    G := gens[i] * Z(2);
    Print("\n======================================================\n");
    Print("Working with Generator Matrix G", i, "\n\n");
    
    # code := GeneratorMatCode(G, GF(2));

    # Print("rank(G)=", RankMat(G), "\n\n");

    # H := CheckMat(code);
    # Print("Original parity check matrix for G", i, "\n");
    # Print("G =\n"); Display(G);
    # Print("H =\n"); Display(H);
    # S := G * TransposedMat(H);
    # Print("Check G*H^T = 0 ? ", S = 0 * S, "\n\n");

    # Standard form conversion
    Gr := MutableCopyMat(G);
    PutStandardForm(Gr,true);
    coder := GeneratorMatCode(Gr,GF(2));
    Hr := CheckMat(coder);

    Print("Standard form Gr=[I|P], Hr=[P^T|I]\n");
    Print("Gr =\n"); Display(Gr);
    Print("Hr =\n"); Display(Hr);
    Sr := Gr * TransposedMat(Hr);
    Print("Check Gr*Hr^T = 0 ? ", Sr = 0 * Sr, "\n");
od;
