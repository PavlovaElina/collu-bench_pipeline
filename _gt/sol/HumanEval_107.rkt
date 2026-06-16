#lang racket
(define (even_odd_palindrome n)
  (define (pal? x) (let ([s (number->string x)]) (string=? s (list->string (reverse (string->list s))))))
  (define pals (filter pal? (range 1 (+ n 1))))
  (list (length (filter even? pals)) (length (filter odd? pals))))
