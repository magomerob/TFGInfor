import extractor
import homology
import topology
import visualization
import os
import time
import cProfile

def main():
    i = 0
    for filename in os.listdir('largedata'):
        print(f"Starting {filename}")
        matrixFiles  = extractor.extract(filename, input_dir='largedata', prompt="MARCIUS:\nThey are dissolved: hang 'em!\nThey said they ", text_dir="text", temperature=0.0)
        for matrixFile in matrixFiles:
            print(f"Processed {filename} -> {matrixFile}")
            homoFile = homology.process_matrix(matrixFile, nThresholds = 20, thresholdMax=0.5)
            print(f"Processed {matrixFile} -> {homoFile}")
            topoFile = topology.analyze_complex(homoFile, dimensions=3, outputdir="topolong")
            print(f"Processed {homoFile} -> {topoFile}")
            #visualFile = visualization.visualize_all(topoFile, output_dir="visualizations")
            #print(f"Processed {topoFile} -> {visualFile}")
            #visualization.visualize_separated(topoFile, output_dir="visualizations-split", figures=[True, True, True])

def test():
    for filename in os.listdir('grande'):
        for i in range(250):
            print(f"Starting {filename} with temperature {i/100}")
            matrixFiles  = extractor.extract(filename, input_dir='grande', prompt="MARCIUS:\nThey are dissolved: hang 'em!\nThey said they ", text_dir="text_temps", temperature=i/100, output_dir="temps", timesblock=2)
            homoFile = homology.process_matrix(matrixFiles[0], nThresholds = 20, thresholdMax=0.5, outputdir="homotemp")
            topoFile = topology.analyze_complex(homoFile, dimensions=3, outputdir="topotemp")
            #visualFile = visualization.visualize_all(topoFile, output_dir="visualizations-temp")
        
if __name__ == "__main__":
    #cProfile.run('main()')
    main()
    #test()
    
