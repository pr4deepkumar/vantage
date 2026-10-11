output "alb_dns_name" {
  description = "Public DNS name of Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR Docker image repository URL"
  value       = aws_ecr_repository.app.repository_url
}

output "rds_endpoint" {
  description = "PostgreSQL RDS Database endpoint"
  value       = aws_db_instance.postgres.endpoint
}
