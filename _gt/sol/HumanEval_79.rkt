#lang racket
(define (decimal_to_binary decimal)
  (string-append "db" (number->string decimal 2) "db"))
