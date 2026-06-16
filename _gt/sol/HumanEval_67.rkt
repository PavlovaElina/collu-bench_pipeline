#lang racket
(define (fruit_distribution s n)
  (define nums (map string->number (filter string->number (string-split s))))
  (- n (apply + nums)))
