resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "${var.app_name}-app-secrets"
  description = "Application secrets for VANTAGE telemetry"
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    DATABASE_URL      = "postgresql://${aws_db_instance.postgres.username}:${var.db_password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
    ANTHROPIC_API_KEY = var.anthropic_api_key
  })
}
