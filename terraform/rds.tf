resource "aws_db_subnet_group" "rds" {
  name       = "${var.app_name}-rds-subnet-group"
  subnet_ids = [aws_subnet.private_1.id, aws_subnet.private_2.id]

  tags = {
    Name = "${var.app_name}-rds-subnet-group"
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = "${var.app_name}-db"
  allocated_storage      = 20
  max_allocated_storage  = 100
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t4g.micro"
  db_name                = "vantage"
  username               = "vantage_admin"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.rds.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  skip_final_snapshot    = true

  tags = {
    Name = "${var.app_name}-db"
  }
}
