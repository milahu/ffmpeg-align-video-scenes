{ pkgs ? import <nixpkgs> { }
  #pkgs ? import ./. {}
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gnumake
    (python3.withPackages (pp: with pp; [
      librosa
      scipy
      matplotlib
      nur.repos.milahu.python3.pkgs.audalign
      nur.repos.milahu.python3.pkgs.scikits-audiolab
      imagehash
      pillow
      opencv4
      # no. too expensive to build
      #(opencv4.override { enableGtk3 = true; })
      nur.repos.milahu.python3.pkgs.decord
    ]))
    # fix: ModuleNotFoundError: No module named 'audalign'
    nur.repos.milahu.python3.pkgs.audalign
    nur.repos.milahu.python3.pkgs.scikits-audiolab
    nur.repos.milahu.python3.pkgs.decord
  ];
}
