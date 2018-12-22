**Universal attacks on equivariant networks**

This code generates universal adversary attacks for RotEqNet and GCNN architectures using top singular vectors 
of input-based attacks provided by FGSM. 


In order to calculate attack directions provided by FGSM for RotEqNet or GCNN architectures
`python main.py --architecture RotEqNet` or `python main.py --architecture GCNN`. First launch will download 
MNIST dataset for FGSM. For further visualization run `python analysis.py`. Pictures are saved into `pic` folder.