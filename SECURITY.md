# Security Policy

## Supported Versions

We actively support the following versions of Research Assistant:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Research Assistant, please report it to us responsibly.

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities
2. Email us at [your-email@domain.com] with:
   - A detailed description of the vulnerability
   - Steps to reproduce the issue
   - Your assessment of the impact
   - Any suggested fixes or mitigations

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Regular Updates**: We will keep you informed of our progress
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days

### Disclosure Policy

- We will coordinate with you on the disclosure timeline
- We will credit you in our security advisory (unless you prefer to remain anonymous)
- We will not take legal action against researchers who report vulnerabilities responsibly

## Security Best Practices

### For Users

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use `.env` files and never share them
3. **Dependencies**: Keep dependencies updated
4. **Access Control**: Limit access to sensitive data
5. **HTTPS**: Always use HTTPS in production

### For Contributors

1. **Code Review**: All code changes require review
2. **Dependencies**: Vet new dependencies for security issues
3. **Input Validation**: Always validate and sanitize user inputs
4. **Error Handling**: Don't expose sensitive information in error messages
5. **Logging**: Avoid logging sensitive information

## Security Features

- **API Key Management**: Secure handling of API credentials
- **Input Sanitization**: Protection against injection attacks
- **Rate Limiting**: Built-in protections against abuse
- **Error Handling**: Secure error messages
- **Dependency Scanning**: Regular security audits

## Contact

For security-related questions or concerns, please contact:
- Email: [your-email@domain.com]
- GPG Key: [optional]

---

Thank you for helping keep Research Assistant secure!
