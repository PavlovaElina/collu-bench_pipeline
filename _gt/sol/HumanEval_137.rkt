#lang racket
(define (compare_one a b)
  (define (val x) (if (string? x) (string->number (string-replace x "," ".")) x))
  (define va (val a)) (define vb (val b))
  (cond [(> va vb) a] [(< va vb) b] [else #f]))
