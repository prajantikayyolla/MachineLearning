import sys
import os

if __name__ == '__main__':
        import paramiko

        # Connect to remote host
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('thunder.cise.ufl.edu', username='prajan', password='Prajan@28')

        # Setup sftp connection and transmit this script
        sftp = client.open_sftp()
        sftp.put(__file__, 'VAE.py')
        sftp.close()

        # Run the transmitted script remotely without args and show its output.
        # SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
        stdout = client.exec_command('python2 VAE.py')[1]
        for line in stdout:
            # Process each line in the remote output
            print(line)

        client.close()
        sys.exit(0)
