#lang racket
(define (largest_divisor n)
  (for/first ([d (in-range (- n 1) 0 -1)] #:when (= 0 (modulo n d))) d))
