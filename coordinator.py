import extractor
import homology
import topology
import visualization
import os
import time
import cProfile

def main():
    for filename in os.listdir('grande'):
        print(f"Starting {filename}")
        matrixFiles  = extractor.extract(filename, input_dir='grande', prompt="BRUTUS:\nThe news is, ", text_dir="text", temperature=0.0)
        for matrixFile in matrixFiles:
            print(f"Processed {filename} -> {matrixFile}")
            homoFile = homology.process_matrix(matrixFile, nThresholds = 500, thresholdMax=0.5)
            print(f"Processed {matrixFile} -> {homoFile}")
            topoFile = topology.analyze_complex(homoFile, dimensions=3)
            print(f"Processed {homoFile} -> {topoFile}")
            visualFile = visualization.visualize_all(topoFile, output_dir="visualizations")
            print(f"Processed {topoFile} -> {visualFile}")
            visualization.visualize_separated(topoFile, output_dir="visualizations-split")

            
if __name__ == "__main__":
    #cProfile.run('main()')
    main()
    
