#lang racket
(define (remove_vowels s)
  (list->string
   (filter (lambda (c) (not (member (char-downcase c) '(#\a #\e #\i #\o #\u))))
           (string->list s))))
