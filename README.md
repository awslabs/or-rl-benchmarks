## OR Rl Benchmarks

The source code for the paper: 'ORL: Reinforcement Learning Benchmarks for Online Stochastic Optimization Problems'

## AWS SageMaker Setup

1. Login to your AWS account.
1. Search for `SageMaker`
1. Click `Notebook instances`
1. Enter notebook name of your choice &rarr; choose instance type `ml.t2.medium` 
1. Create a new IAM role if required. Also attach `AmazonEC2ContainerRegistryPowerUser` policy to the role. See [guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_manage-attach-detach.html#add-policies-console) for adding permissions to an IAM Role.
1. Under `Git repositories`, click `Additional repository` &rarr; Select `Clone a Public repo` and enter `https://github.com/awslabs/or-rl-benchmarks.git`
1. Click `Create Notebook` and wait for the instance to come up.
1. Once the instance is ready, select it and click `Open Jupyter` 
1. Above step will load this repository and then you can follow the python notebooks for each problem.

## Contributors
This is a joint work of - 

Bharathan Balaji @bbalaji-ucsd

Jordan Bell-Masterson 

Enes Bilgin 

Andreas Damianou @adamian

Pablo Moreno Garcia

Arpit Jain @arpitjain2811

Runfei Luo @annaluo676

Alvaro Maggiar

Balakrishnan Narayanaswamy

Chun Ye

## License Summary

This sample code is made available under the MIT-0 license. See the LICENSE file.
